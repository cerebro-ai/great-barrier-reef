import re
import subprocess


def get_cpu_temp():
    """Gets the main cpu temperature with 'sensors' and returns it as a float"""

    command = "sensors"
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, shell=True) as process:
        output = process.communicate()[0].decode("utf-8")
    lines = output.splitlines()
    line = [x for x in lines if "Package" in x][0]
    m = re.search(r"(?<=\+)\d{2,3}\.\d(?=°C)", line).group(0)
    return float(m)