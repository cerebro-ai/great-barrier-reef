import subprocess
import time
from pathlib import Path
import datetime

while True:
    date = datetime.datetime.now()
    command = "sensors"
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, shell=True) as process:
        output = process.communicate()[0].decode("utf-8")
    lines = output.splitlines()
    line = [x for x in lines if "Package" in x]
    log = f"{date}: {line[0]}"
    print(log)
    with Path("temp_log.txt").open("a") as f:
        f.write(log + "\n")
    time.sleep(15)
