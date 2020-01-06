# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import

import os
import threading

from collections import defaultdict

import six

from horovod.run.common.util import hosts, safe_shell_exec, timeout


READY = 'READY'
SUCCESS = 'SUCCESS'
FAILURE = 'FAILURE'

DISCOVER_HOSTS_FREQUENCY_SECS = 1.0


class Workers(object):
    def __init__(self, driver):
        self._driver = driver
        self._lock = threading.Lock()
        self._states = {}
        self._workers = defaultdict(set)
        self._barrier = None
        self._rendezvous_id = 0

    def get(self, state):
        return self._workers[state]

    def count(self, state):
        return len(self._workers[state])

    def reset(self, size):
        with self._lock:
            self._states.clear()
            self._workers.clear()
            self._barrier = threading.Barrier(parties=size, action=self._action)
            self._rendezvous_id += 1

    def last_rendezvous(self):
        return self._rendezvous_id

    def record_ready(self, host, slot):
        return self._record_state(host, slot, READY)

    def record_success(self, host, slot):
        return self._record_state(host, slot, SUCCESS)

    def record_failure(self, host, slot):
        return self._record_state(host, slot, FAILURE)

    def _record_state(self, host, slot, state):
        if self._driver.finished():
            return self._rendezvous_id

        key = (host, slot)
        with self._lock:
            if key in self._states:
                self._barrier.reset()
            self._states[key] = state
            self._workers[state].add(key)
            rendezvous_id = self._rendezvous_id

        rendezvous_id = self._wait(key, state, rendezvous_id)
        return rendezvous_id

    def _wait(self, key, state, rendezvous_id):
        while True:
            try:
                self._barrier.wait()
                return rendezvous_id
            except threading.BrokenBarrierError:
                if self._barrier.broken:
                    # Timeout or other non-recoverable error, so exit
                    raise

                with self._lock:
                    rendezvous_id = self._rendezvous_id
                    saved_state = self._states.get(key, state)
                    if saved_state != state:
                        raise RuntimeError('State {} overridden by {}'.format(state, saved_state))

    def _action(self):
        self._driver.on_workers_recorded()


class Host(object):
    def __init__(self):
        self._event = threading.Event()

        # TODO(travis): blacklisted hosts should have a timeout period that increases with each failure
        self._blacklisted = False

    def get_event(self):
        if self._event.is_set():
            event = threading.Event()
            self._event = event
        return self._event

    def set_event(self):
        self._event.set()

    def blacklist(self):
        self._blacklisted = True
        self.set_event()

    def is_blacklisted(self):
        return self._blacklisted


class Results(object):
    def __init__(self, driver):
        self._driver = driver
        self._results = {}
        self._wait_cond = threading.Condition()

    def add_result(self, key, value):
        self._wait_cond.acquire()
        try:
            if key in self._results:
                return

            self._results[key] = value
            if len(self._results) == self._driver.world_size():
                self._wait_cond.notify_all()
        finally:
            self._wait_cond.release()

    def get_results(self):
        self._wait_cond.acquire()
        try:
            while len(self._results) < self._driver.world_size():
                self._wait_cond.wait()
            return self._results
        finally:
            self._wait_cond.release()


class ElasticDriver(object):
    def __init__(self, discovery_script, min_np, max_np, slots, start_timeout=None):
        self._discovery_script = discovery_script
        self._min_np = min_np
        self._max_np = max_np
        self._slots = slots

        self._available_hosts = set()
        self._available_slots = {}

        self._assigned_hosts = []
        self._host_assignments = {}
        self._world_size = 0

        self._wait_hosts_cond = threading.Condition()
        self._start_timeout = start_timeout or int(os.getenv('HOROVOD_ELASTIC_START_TIMEOUT', '600'))

        self._create_worker_fn = None
        self._hosts = defaultdict(Host)

        self._workers = Workers(self)
        self._results = Results(self)
        self._shutdown = threading.Event()

        self._discovery_thread = threading.Thread(target=self._discover_hosts)
        self._discovery_thread.daemon = True
        self._discovery_thread.start()

    def start(self, np, create_worker_fn):
        self._create_worker_fn = create_worker_fn
        self._activate_hosts(np)

    def wait_for_available_hosts(self, min_np):
        tmout = timeout.Timeout(
            self._start_timeout,
            message='Timed out waiting for {{activity}}. Please check that you have '
                    'enough resources to run at least {min_np} Horovod processes.'.format(min_np=min_np))

        self._wait_hosts_cond.acquire()
        try:
            while self._count_available_slots() < min_np:
                self._wait_hosts_cond.wait(tmout.remaining())
                tmout.check_time_out_for('minimum number of hosts to become available')
        finally:
            self._wait_hosts_cond.release()

    def get_results(self):
        return self._results.get_results()

    def stop(self):
        self._shutdown.set()

    def finished(self):
        return self._shutdown.is_set()

    def record_ready(self, host, slot):
        self._workers.record_ready(host, slot)

    def on_workers_recorded(self):
        # Check for success state, if any process succeeded, shutdown all other processes
        if self._workers.count(SUCCESS) > 0:
            self.stop()
            return

        # Check that all processes failed, indicating that processing should stop
        if self._workers.count(FAILURE) == self._world_size:
            self.stop()
            return

        # Check for failures, and add them to the blacklisted hosts list
        failures = self._workers.get(FAILURE)
        for host, slot in failures:
            self._hosts[host].blacklist()

        self._activate_hosts(self._min_np)

    def world_size(self):
        return self._world_size

    def local_size(self, host):
        return len(self._host_assignments[host])

    def get_slot_info(self, host, slot):
        return self._host_assignments[host][slot]

    def get_available_hosts(self):
        return list(self._available_hosts)

    def _activate_hosts(self, min_np):
        self.wait_for_available_hosts(min_np)
        new_assigned_hosts = self._update_assigned_hosts()
        self._workers.reset(self.world_size())
        for host in new_assigned_hosts:
            self._start_worker_processes(host)

    def _discover_hosts(self):
        while not self._shutdown.is_set():
            self._wait_hosts_cond.acquire()
            try:
                if self._update_available_hosts():
                    self._wait_hosts_cond.notify_all()
            finally:
                self._wait_hosts_cond.release()
            self._shutdown.wait(DISCOVER_HOSTS_FREQUENCY_SECS)

    def _update_available_hosts(self):
        prev_hosts = self._available_hosts
        prev_slots = self._available_slots
        self._available_hosts, self._available_slots = self._find_available_hosts_and_slots()
        return prev_hosts != self._available_hosts or prev_slots != self._available_slots

    def _find_available_hosts_and_slots(self):
        stdout = six.StringIO()
        exit_code = safe_shell_exec.execute(self._discovery_script, stdout=stdout)
        if exit_code != 0:
            raise RuntimeError('Failed to execute discovery script: {}. Exit code: {}'
                               .format(self._discovery_script, exit_code))

        availabe_hosts = set()
        available_slots = {}
        hosts_and_slots = set(stdout.getvalue().strip().split('\n'))
        for line in hosts_and_slots:
            host = line
            if ':' in line:
                host, slots = line.split(':')
                available_slots[host] = int(slots)
            availabe_hosts.add(host)
        return availabe_hosts, available_slots

    def _update_assigned_hosts(self):
        new_assigned_hosts = []
        self._assigned_hosts = [host for host in self._assigned_hosts
                                if host in self._available_hosts and not self._hosts[host].is_blacklisted()]
        current_hosts = set(self._assigned_hosts)
        for host in self._available_hosts:
            if host not in current_hosts and not self._hosts[host].is_blacklisted():
                new_assigned_hosts.append(host)
                self._assigned_hosts.append(host)
        self._update_host_assignments()
        return new_assigned_hosts

    def _update_host_assignments(self):
        host_list = [hosts.HostInfo(host, self._get_slots(host)) for host in self._assigned_hosts]
        host_assignments_list = hosts.get_host_assignments(host_list, self._min_np, self._max_np)
        host_assignments = defaultdict(list)
        for slot_info in host_assignments_list:
            host_assignments[slot_info.hostname].append(slot_info)
        self._host_assignments = host_assignments
        self._world_size = len(host_assignments_list)

    def _count_available_slots(self):
        return sum([self._get_slots(host) for host in self._available_hosts])

    def _get_slots(self, host):
        if host in self._available_slots:
            return self._available_slots[host]
        return self._slots

    def _start_worker_processes(self, host):
        for slot_info in self._host_assignments[host]:
            self._start_worker_process(slot_info)

    def _start_worker_process(self, slot_info):
        create_worker_fn = self._create_worker_fn
        host_event = self._hosts[slot_info.hostname].get_event()

        def run_worker():
            res = create_worker_fn(slot_info, [host_event])
            exit_code, timestamp = res
            self._handle_worker_exit(slot_info, exit_code, timestamp)

        thread = threading.Thread(target=run_worker)
        thread.daemon = True
        thread.start()

    def _handle_worker_exit(self, slot_info, exit_code, timestamp):
        if self._hosts[slot_info.hostname].is_blacklisted():
            # Ignore blacklisted hosts
            return

        if exit_code == 0:
            rendezvous_id = self._workers.record_success(slot_info.hostname, slot_info.local_rank)
        else:
            rendezvous_id = self._workers.record_failure(slot_info.hostname, slot_info.local_rank)

        if self.finished() and self._workers.last_rendezvous() == rendezvous_id:
            name = '{}[{}]'.format(slot_info.hostname, slot_info.local_rank)
            self._results.add_result(name, (exit_code, timestamp))
