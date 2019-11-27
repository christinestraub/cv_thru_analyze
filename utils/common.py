import os
import signal
import subprocess
import utils.logger as logger
import json

_cur_dir = os.path.dirname(os.path.realpath(__file__))


def load_json(json_path):
    with open(json_path, 'r') as jp:
        obj = json.load(jp)
        return obj


def save_json(json_path, obj):
    with open(json_path, 'w') as jp:
        json.dump(obj, jp)
        return True


def check_running_proc(proc_name):
    """
    Check if a process is running or not
    :param proc_name:
    :return:
    """
    try:
        if len(os.popen("ps -aef | grep -i '%s' "
                        "| grep -v 'grep' | awk '{ print $3 }'" % proc_name).read().strip().splitlines()) > 0:
            return True
    except Exception as e:
        logger.error('Failed to get status of the process({}) - {}'.format(proc_name, e))
    return False


def kill_process_by_name(proc_name):
    """
    Kill process by its name
    :param proc_name:
    :return:
    """
    p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    out, err = p.communicate()
    for line in out.decode().splitlines():
        if proc_name in line:
            pid = int(line.split(None, 1)[0])
            print('Found PID({}) of `{}`, killing...'.format(pid, proc_name))
            os.kill(pid, signal.SIGKILL)
