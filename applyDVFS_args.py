import subprocess
import sys
import json

CoreLevel = 7
MemoryLevel = 3
CoreFrequencyInput = int(sys.argv[1])
CoreVoltageInput = int(sys.argv[2])

CoreFreq = [853, 860, 870, 880, 890, 900, 910, CoreFrequencyInput]
CoreVoltage = [800, 810, 820, 830, 840, 850, 860, CoreVoltageInput]
MemoryFreq = [168, 400, 450, 1000]
MemoryVoltage = [801, 805, 810, 1001]

def runBashCommand(bashCommand):
    """Runs a bash command

    Args:
        bashCommand: Command to be run

    Returns:
        The resulting process
    """
    # print("Running %s" % (bashCommand))
    seconds = 10
    try:
        if "3.6" in sys.version:
            process = subprocess.run(bashCommand.split(),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 stdin=subprocess.PIPE,
                                 check=True,
                                 timeout=seconds)
            return process.stdout.decode("utf-8") 
        else:
            process = subprocess.run(bashCommand.split(),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 check=True,
                                 text=True,
                                 timeout=seconds)
            return process.stdout
    except subprocess.CalledProcessError as e:
        print()
        print("\tERROR: Execution warmup ", str(e.returncode))
        print()
    except subprocess.TimeoutExpired as t:
        print()
        print("\tERROR: Timeout warmup")
        print()
    sys.exit(-1)

def preDVFSconfig():
    # Enables Overdrive
    result = runBashCommand("rocm-smi --setfan 255")
    if "Successfully" not in result:
        print("Not able to reset GPU")
        sys.exit(-1)

    # Enables Overdrive
    result = runBashCommand("rocm-smi --autorespond y --setoverdrive 20")
    if "Successfully" not in result:
        print("Not able to reset GPU")
        sys.exit(-1)

    result = runBashCommand("rocm-smi --autorespond y --setpoweroverdrive 300")
    if "Successfully" not in result:
        print("Not able to reset GPU")
        sys.exit(-1)

    # Set Core performance level to the one to be tested
    if setPerformanceLevel("core", 0) == False:
        print(" Not able to select core level.")
        return False
    # Set Memory performance level to the highest
    if setPerformanceLevel("mem", 0) == False:
        print(" Not able to select memory level.")
        return False

    return True

def runDVFSscript(command):
    """Runs the DVFS script

    Args:
        command: Command to be run

    Returns:
        The resulting process
    """
    script = "./DVFS " + str(command)
    process = runBashCommand(script)
    return process

def setPerformanceLevel(source, level):
    """Sets a given performance level for the GPU Core and Memory.

    Args:
        source: string containing word "core" or "mem"
        level: an integer between 0-7 for core and 0-3 memory

    Returns:
        True - if action is sucessful.
        False - not possible to apply configuration.
    """
    if source == "core":
        assert level in list(range(
            0, 8)), "Core Performance Level betwen 0 and 7."
        result = runDVFSscript("-P " + str(level))
        if "ERROR" in result:
            return False
    elif source == "mem":
        assert level in list(range(
            0, 4)), "Core Performance Level betwen 0 and 3."
        result = runDVFSscript("-p " + str(level))
        if "ERROR" in result:
            return False
    else:
        print("Not valid source used - core or mem")
        return False

    return True

def editPerformanceLevel(source, level, frequency, voltage):
    """Edits a given performance level for the GPU Core and Memory.

    Args:
        source: string containing word "core" or "mem"
        level: an integer between 0-7 for core and 0-3 memory
        frequency: an integer indicating the frequency
        voltage: an integer indicating the voltage

    Returns:
        True - if action is sucessful.
        False - not possible to apply configuration.
    """
    if source == "core":
        assert level in list(range(
            0, 8)), "Core Performance Level betwen 0 and 7."
        result = runDVFSscript("-L " + str(level) + " -F " + str(frequency) +
                               " -V " + str(voltage))
        if "ERROR" in result:
            return False
    elif source == "mem":
        assert level in list(range(
            0, 4)), "Core Performance Level betwen 0 and 3."
        result = runDVFSscript("-l " + str(level) + " -f " + str(frequency) +
                               " -v " + str(voltage))
        if "ERROR" in result:
            return False
    else:
        print("Not valid source used - core or mem")
        return False

    return True

def editAllPerformanceLevels():
    for level in range(0, 4):
        result = editPerformanceLevel("mem", level, MemoryFreq[level], MemoryVoltage[level])
        if result == False:
            return False
    for level in range(0, 8):
        result = editPerformanceLevel("core", level, CoreFreq[level], CoreVoltage[level])
        if result == False:
            return False

    return True



def currentPerfLevel():
    """Gets the current applied performance level for the Core and Memory

    Returns:
        Tuple on the form (core, memory) indicating the current 
        performance level of the two domains
    """
    global CoreFreq
    global MemoryFreq
    result = runBashCommand("rocm-smi")
    core = -1
    mem = -1
    line = result.split('\n')
    line = line[5].split(" ")
    # Find indices of Core and Mem frequency
    indices = [i for i, s in enumerate(line) if 'Mhz' in s]
    core = line[indices[0]].replace("Mhz", '')
    mem = line[indices[1]].replace("Mhz", '')

    print(core + "," + mem)

    return CoreFreq.index(int(core)), MemoryFreq.index(int(mem))

def currentVoltageIsRespected(currentVoltCore, currentVoltMemory):
    """Gets the current voltage applied to the GPU Core

    Args:
        currentVolt - a path to a filepath.
    Returns:
        True if the current applied voltage respects the
        intended one.
    """

    result = runBashCommand("rocm-smi --showvoltage --json")
    # Find core voltage
    try:
        volt= json.loads(result)["card1"]["Voltage (mV)"]
    except:
        print("Not able to get voltage")
        return False, -1

    if abs(int(volt) - int(currentVoltCore)) <= 5:
        return True, volt

    if abs(int(volt) - int(currentVoltMemory)) <= 5:
        return True, volt

    return False, volt

if preDVFSconfig() == False:
    print("not able to update table for current run")
    sys.exit(-1)
if editAllPerformanceLevels() == False:
    print("not able to update table for current run")
    sys.exit(-1)

# Set Core performance level to the one to be tested
if setPerformanceLevel("core", CoreLevel) == False:
    print(" Not able to select core level.")
    sys.exit(-1)
# Set Memory performance level to the highest
if setPerformanceLevel("mem", MemoryLevel) == False:
    print(" Not able to select memory level.")
    sys.exit(-1)

# Run warm up DVFS program
runBashCommand("./warmup")

# Get current DVFS settings - to make sure it was correctly applyed
cur = currentPerfLevel()
print(cur)
print("%s, %s" % (CoreVoltage[cur[0]], MemoryVoltage[cur[1]]))
if cur != (CoreLevel, MemoryLevel):
    print(" Selected Performance Levels don't match current ones. %s != (%d, %d)" % (cur, CoreLevel, MemoryLevel))
    sys.exit(-1)


    
# Checks if the intended voltage is correctly applied to the GPU
result, volt = currentVoltageIsRespected(CoreVoltage[CoreLevel], MemoryVoltage[MemoryLevel])
print(result, volt)
if result == False:
    print("Current voltage is %d != Core: %d | Memory: %d" % (int(volt), int(CoreVoltage[CoreLevel]), int(MemoryVoltage[MemoryLevel])))
    sys.exit(-1)

sys.exit(0)