# Copyright (c) 2021 Red Hat
#
# This code is part of Ansible, but is an independent component.
# This particular file snippet, and this file snippet only, is BSD licensed.
# Modules you write using this snippet, which is embedded dynamically by Ansible
# still belong to the author of the module, and may assign their own license
# to the complete work.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice,
#      this list of conditions and the following disclaimer in the documentation
#      and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import argparse
import asyncio

# py38 only, See: https://github.com/PyCQA/pylint/issues/2976
import collections  # pylint: disable=syntax-error
import importlib

# py38 only, See: https://github.com/PyCQA/pylint/issues/2976
import inspect  # pylint: disable=syntax-error
import io
import json
import os
import pickle
import signal
import sys
import traceback
import uuid
import zipfile
from datetime import datetime
from zipimport import zipimporter

sys_path_lock = None
env_lock = None

import ansible.module_utils.basic

please_include_me = "bar"


def fork_process():
    """
    This function performs the double fork process to detach from the
    parent process and execute.
    """
    with open("/tmp/turbo-server-logs.txt", "a+") as fd:
        fd.write("\nRunning common.plugins.module_utils.turbo.server.fork_process()\n")
    pid = os.fork()

    if pid == 0:
        fd = os.open(os.devnull, os.O_RDWR)

        # clone stdin/out/err
        for num in range(3):
            if fd != num:
                os.dup2(fd, num)

        if fd not in range(3):
            os.close(fd)

        pid = os.fork()
        if pid > 0:
            os._exit(0)

        # get new process session and detach
        sid = os.setsid()
        if sid == -1:
            raise Exception("Unable to detach session while daemonizing")

        # avoid possible problems with cwd being removed
        os.chdir("/")

        pid = os.fork()
        if pid > 0:
            sys.exit(0)  # pylint: disable=ansible-bad-function
    else:
        sys.exit(0)  # pylint: disable=ansible-bad-function
    with open("/tmp/turbo-server-logs.txt", "a+") as fd:
        fd.write(f"  common.plugins.module_utils.turbo.server.fork_process(): Returning forked process pid: {pid}\n")
    return pid


class EmbeddedModule:
    def __init__(self, ansiblez_path, params):
        self.ansiblez_path = ansiblez_path
        self.collection_name, self.module_name = self.find_module_name()
        self.params = params
        self.module_class = None
        self.debug_mode = False
        self.module_path = (
            "ansible_collections.{collection_name}." "plugins.modules.{module_name}"
        ).format(collection_name=self.collection_name, module_name=self.module_name)

    def find_module_name(self):
        with zipfile.ZipFile(self.ansiblez_path) as zip:
            for path in zip.namelist():
                if not path.startswith("ansible_collections"):
                    continue
                if not path.endswith(".py"):
                    continue
                if path.endswith("__init__.py"):
                    continue
                splitted = path.split("/")
                if len(splitted) != 6:
                    continue
                if splitted[-3:-1] != ["plugins", "modules"]:
                    continue
                collection = ".".join(splitted[1:3])
                name = splitted[-1][:-3]
                return collection, name
        raise Exception("Cannot find module name")

    async def load(self):
        async with sys_path_lock:
            # Add the Ansiblez_path in sys.path
            sys.path.insert(0, self.ansiblez_path)

            # resettle the loaded modules that were associated
            # with a different Ansiblez.
            for path, module in sorted(tuple(sys.modules.items())):
                if path and module and path.startswith("ansible_collections"):
                    try:
                        prefix = sys.modules[path].__loader__.prefix
                    except AttributeError:
                        # Not from a zipimporter loader, skipping
                        continue
                    # Reload package modules only, to pick up new modules from
                    # packages that have been previously loaded.
                    if hasattr(sys.modules[path], "__path__"):
                        py_path = self.ansiblez_path + os.sep + prefix
                        my_loader = zipimporter(py_path)
                        sys.modules[path].__loader__ = my_loader
                        try:
                            importlib.reload(sys.modules[path])
                        except ModuleNotFoundError:
                            pass
            # Finally, load the plugin class.
            self.module_class = importlib.import_module(self.module_path)

    async def unload(self):
        async with sys_path_lock:
            sys.path = [i for i in sys.path if i != self.ansiblez_path]

    def create_profiler(self):
        if self.debug_mode:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()
            return pr

    def print_profiling_info(self, pr):
        if self.debug_mode:
            import pstats

            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(pr).sort_stats(sortby)
            ps.print_stats(20)

    def print_backtrace(self, backtrace):
        if self.debug_mode:
            print(backtrace)  # pylint: disable=ansible-bad-function

    async def run(self):
        class FakeStdin:
            buffer = None

        from .exceptions import (
            EmbeddedModuleFailure,
            EmbeddedModuleSuccess,
            EmbeddedModuleUnexpectedFailure,
        )

        # monkeypatching to pass the argument to the module, this is not
        # really safe, and in the future, this will prevent us to run several
        # modules in parallel. We can maybe use a scoped monkeypatch instead
        _fake_stdin = FakeStdin()
        _fake_stdin.buffer = io.BytesIO(self.params.encode())
        sys.stdin = _fake_stdin
        # Trick to be sure ansible.module_utils.basic._load_params() won't
        # try to build the module parameters from the daemon arguments
        sys.argv = sys.argv[:1]
        ansible.module_utils.basic._ANSIBLE_ARGS = None
        pr = self.create_profiler()
        if not hasattr(self.module_class, "main"):
            raise EmbeddedModuleFailure("No main() found!")
        try:
            if inspect.iscoroutinefunction(self.module_class.main):
                await self.module_class.main()
            elif pr:
                pr.runcall(self.module_class.main)
            else:
                self.module_class.main()
        except EmbeddedModuleSuccess as e:
            self.print_profiling_info(pr)
            return e.kwargs
        except EmbeddedModuleFailure as e:
            backtrace = traceback.format_exc()
            self.print_backtrace(backtrace)
            raise
        except Exception as e:
            backtrace = traceback.format_exc()
            self.print_backtrace(backtrace)
            raise EmbeddedModuleUnexpectedFailure(str(backtrace))
        else:
            raise EmbeddedModuleUnexpectedFailure(
                "Likely a bug: exit_json() or fail_json() should be called during the module excution"
            )


async def run_as_lookup_plugin(data):
    with open("/tmp/turbo-server-logs.txt", "a+") as fd:
        fd.write("\nRunning common.plugins.module_utils.turbo.server.run_as_lookup_plugin()\n")
    errors = None
    result = None

    try:
        import ansible.plugins.loader as plugin_loader
        from ansible.module_utils._text import to_native
        from ansible.parsing.dataloader import DataLoader
        from ansible.template import Templar
    except ImportError as e:
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write(f"  common.plugins.module_utils.turbo.server.run_as_lookup_plugin(): ImportError: {e}\n")
        errors = str(e)
        return [result, errors]

    try:
        (
            lookup_name,
            terms,
            variables,
            kwargs,
        ) = data

        # load lookup plugin
        templar = Templar(loader=DataLoader(), variables=None)
        ansible_collections = "ansible_collections."
        if lookup_name.startswith(ansible_collections):
            lookup_name = lookup_name.replace(ansible_collections, "", 1)
        ansible_plugins_lookup = ".plugins.lookup."
        if ansible_plugins_lookup in lookup_name:
            lookup_name = lookup_name.replace(ansible_plugins_lookup, ".", 1)

        instance = plugin_loader.lookup_loader.get(name=lookup_name, loader=templar._loader, templar=templar)

        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write(
                f"  common.plugins.module_utils.turbo.server.run_as_lookup_plugin(): Loaded plugin named: {lookup_name}\n"
            )

        if not hasattr(instance, "_run"):
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write(
                    f"  common.plugins.module_utils.turbo.server.run_as_lookup_plugin(): Plugin  {lookup_name} has no `_run()` function, returning [None, 'No _run() found']\n"
                )
            return [None, "No _run() found"]
        if inspect.iscoroutinefunction(instance._run):
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write(f"  common.plugins.module_utils.turbo.server.run_as_lookup_plugin(): Awaiting {lookup_name}._run()\n")
            result = await instance._run(terms, variables=variables, **kwargs)
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write(f"  common.plugins.module_utils.turbo.server.run_as_lookup_plugin(): Continuing after awaiting {lookup_name}._run()\n")
        else:
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write(f"\n  common.plugins.module_utils.turbo.server.run_as_lookup_plugin(): Calling {lookup_name}._run()\n")
            result = instance._run(terms, variables=variables, **kwargs)
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write(f"\n  common.plugins.module_utils.turbo.server.run_as_lookup_plugin(): Continuing after calling {lookup_name}._run()\n")
    except Exception as e:
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write(f"\n  common.plugins.module_utils.turbo.server.run_as_lookup_plugin(): Encountered exception: {e}\n")
        errors = to_native(e)

    with open("/tmp/turbo-server-logs.txt", "a+") as fd:
        fd.write(f"  common.plugins.module_utils.turbo.server.run_as_lookup_plugin(): Returning {[result, errors]}\n")

    return [result, errors]


async def run_as_module(content, debug_mode):
    result = None
    from ansible_collections.cloud.common.plugins.module_utils.turbo.exceptions import (
        EmbeddedModuleFailure,
    )

    try:
        (
            ansiblez_path,
            params,
            env,
        ) = json.loads(content)
        if debug_mode:
            print(  # pylint: disable=ansible-bad-function
                f"-----\nrunning {ansiblez_path} with params: ¨{params}¨"
            )

        embedded_module = EmbeddedModule(ansiblez_path, params)
        if debug_mode:
            embedded_module.debug_mode = True

        await embedded_module.load()
        try:
            async with env_lock:
                os.environ.clear()
                os.environ.update(env)
                result = await embedded_module.run()
        except SystemExit:
            backtrace = traceback.format_exc()
            result = {"msg": str(backtrace), "failed": True}
        except EmbeddedModuleFailure as e:
            result = {"msg": str(e), "failed": True}
            if e.kwargs:
                result.update(e.kwargs)
        except Exception as e:
            result = {
                "msg": traceback.format_stack() + [str(e)],
                "failed": True,
            }
        await embedded_module.unload()
    except Exception as e:
        result = {"msg": traceback.format_stack() + [str(e)], "failed": True}
    return result


class AnsibleVMwareTurboMode:
    def __init__(self):
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write("\nInitializing common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode as server\n")
        self.sessions = collections.defaultdict(dict)
        self.socket_path = None
        self.ttl = None
        self.debug_mode = None
        self.jobs_ongoing = {}
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write("  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode: Initialized, returning to caller\n")

    async def ghost_killer(self):
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write("\nRunning common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.ghost_killer()\n")
        while True:
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write(f"  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.ghost_killer() awaiting sleep({self.ttl})\n")
            await asyncio.sleep(self.ttl)
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write("  \ncommon.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.ghost_killer() continuing after await\n")
            running_jobs = {job_id: start_date for job_id, start_date in self.jobs_ongoing.items() if (datetime.now() - start_date).total_seconds() < 10}
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write(f"  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.ghost_killer(): running jobs: {running_jobs}\n")
            if running_jobs:
                with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                    fd.write("  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.ghost_killer(): running while True loop again\n")
                continue
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write(
                    "  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.ghost_killer(): no running jobs, calling AnsibleVMwareTurboMode.stop()\n"
                )
            self.stop()

    async def handle(self, reader, writer):
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write(
                "\nRunning common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle() with args:\n"
                f"  reader: {reader}\n"
                f"  writer: {writer}\n"
                f"\nCurrent process pid: {os.getpid()}\n"
            )

        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write(
                f"  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.start(): Thread ID: {self.loop._thread_id}\n"
                f"  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.start(): PID: {os.getpid()}\n"
            )

        def _terminate(result, plugin_type):
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write("\nRunning internatl common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle()._terminate()\n")
            try:
                response = json.dumps(result).encode()

            except Exception as e:
                error = None
                message = f"Exception encountered parsing result {str(result)}. Error: {traceback.format_stack() + [str(e)]}"
                if plugin_type == "module":
                    error = {"msg": message, "failed": True}
                elif plugin_type == "lookup":
                    error = [[""], message]
                response = json.dumps(error).encode()

            writer.write(response)
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write(
                    f"  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle()._terminate(): Wrote result to reader: {json.dumps(result)}\n"
                )
            writer.close()
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write("  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle()._terminate(): Writer closed\n")

        result = None
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write("  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle(): Canceling running ghost_killer() task\n")
        self._watcher.cancel()
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write(
                "  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle(): Creating loop task common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.ghost_killer()\n"
            )
        self._watcher = self.loop.create_task(self.ghost_killer())
        job_id = str(uuid.uuid4())
        self.jobs_ongoing[job_id] = datetime.now()
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write(
                f"  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle(): Added job_id {job_id} to ongoing jobs\n"
                f"  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle(): Awaiting reader.read()\n"
            )

        raw_data = await reader.read()

        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write("  \ncommon.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle() Continuing after awaiting reader.read()\n")

        if not raw_data:
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write("  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle(): No raw data, returning\n")
            return

        (plugin_type, content) = pickle.loads(raw_data)
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write(
                f"  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle(): plugin_type: {plugin_type}\n"
            )

        if plugin_type == "module":
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write(
                    "  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle(): Awaiting common.plugins.module_utils.turbo.server.run_as_module()\n"
                )
            try:
                result = await run_as_module(content, debug_mode=self.debug_mode)
                with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                    fd.write(
                        "\n  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle(): Continuing after awaiting common.plugins.module_utils.turbo.server.run_as_module()\n"
                    )
            except Exception as e:
                _terminate({"msg": f"Uncaught exception while trying to run module in TurboMode. Error: {e}. Stacktrace: {traceback.format_stack() + [str(e)]}", "failed": True}, plugin_type)

        elif plugin_type == "lookup":
            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write(
                    "  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle(): Awaiting common.plugins.module_utils.turbo.server.run_as_lookup_plugin()\n"
                )
            try:
                result = await run_as_lookup_plugin(content)
                with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                    fd.write(
                        "\n  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle(): Continuing after awaiting common.plugins.module_utils.turbo.server.run_as_lookup_plugin()\n"
                    )
            except Exception as e:
                with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                    fd.write("  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle(): Calling internal _terminate()\n")

                _terminate([None, f"Uncaught exception while trying to run lookup plugin in TurboMode. Error: {e}. Stacktrace: {traceback.format_stack() + [str(e)]}"], plugin_type)

        _terminate(result, plugin_type)
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write("\n  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle(): Continuing after calling internal _terminate()\n")
        del self.jobs_ongoing[job_id]
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write(f"  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle(): Deleted job {job_id} from ongoing jobs\n")

    def handle_exception(self, loop, context):
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write("\nRunning common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle_exception()\n")
        e = context.get("exception")
        traceback.print_exception(type(e), e, e.__traceback__)
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write(f"  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle_exception(): Exception: {e}\n")
            fd.write(
                "  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle_exception(): Calling common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.stop()\n"
            )
        self.stop()

    def start(self):
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write("\nRunning common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.start()\n")
        self.loop = asyncio.get_event_loop()
        self.loop.add_signal_handler(signal.SIGTERM, self.stop)
        self.loop.set_exception_handler(self.handle_exception)
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write(
                "  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.start(): Creating loop task common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.ghost_killer()\n"
            )
        self._watcher = self.loop.create_task(self.ghost_killer())

        import sys

        try:
            from ansible.plugins.loader import init_plugin_loader

            with open("/tmp/turbo-server-logs.txt", "a+") as fd:
                fd.write("  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.start(): Calling ansible.plugins.loader.init_plugin_loader()\n")
            init_plugin_loader()
        except ImportError:
            # Running on Ansible < 2.15
            pass

        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write(
                "  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.start(): Creating loop task asyncio.start_unix_server with callable common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.handle()\n"
            )
        if sys.hexversion >= 0x30A00B1:
            # py3.10 drops the loop argument of create_task.
            self.loop.create_task(
                asyncio.start_unix_server(self.handle, path=self.socket_path)
            )
        else:
            self.loop.create_task(
                asyncio.start_unix_server(
                    self.handle, path=self.socket_path, loop=self.loop
                )
            )
        self.loop.run_forever()

    def stop(self):
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write("\nRunning common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.stop()\n")
        os.unlink(self.socket_path)
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write("  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.stop(): socket path unlinked\n")
        self.loop.stop()
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write("  common.plugins.module_utils.turbo.server.AnsibleVMwareTurboMode.stop(): loop stopped\n")


if __name__ == "__main__":
    with open("/tmp/turbo-server-logs.txt", "w+") as fd:
        fd.write("Running common.plugins.module_utils.turbo.server().main()\n")
    parser = argparse.ArgumentParser(description="Start a background daemon.")
    parser.add_argument("--socket-path")
    parser.add_argument("--ttl", default=15, type=int)
    parser.add_argument("--fork", action="store_true")

    args = parser.parse_args()
    if args.fork:
        with open("/tmp/turbo-server-logs.txt", "a+") as fd:
            fd.write("  common.plugins.module_utils.turbo.server().main: Calling common.plugins.module_utils.turbo.server().fork_process()\n")
        fork_process()
    sys_path_lock = asyncio.Lock()
    env_lock = asyncio.Lock()

    with open("/tmp/turbo-server-logs.txt", "a+") as fd:
        fd.write("  \ncommon.plugins.module_utils.turbo.server().main: Initializing AnsibleVMwareTurboMode()\n")
    server = AnsibleVMwareTurboMode()
    server.socket_path = args.socket_path
    server.ttl = args.ttl
    server.debug_mode = True
    with open("/tmp/turbo-server-logs.txt", "a+") as fd:
        fd.write(
            "\n  common.plugins.module_utils.turbo.server().main: AnsibleVMwareTurboMode() attributes updated to:\n"
            f"    socket_path: {server.socket_path}\n"
            f"    ttl: {server.ttl}\n"
            f"    debug_mode: {server.debug_mode}\n"
        )
    with open("/tmp/turbo-server-logs.txt", "a+") as fd:
        fd.write("  common.plugins.module_utils.turbo.server().main: Calling AnsibleVMWareTurboMode.start()\n")
    server.start()
