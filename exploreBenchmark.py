import argparse
import subprocess
from pathlib import Path
import os
import re
import sys
import json
import os
import time

CoreFreq = []
CoreVoltage = []
MemoryFreq = []
MemoryVoltage = []

def removeFile(file):
    # Remove the file
    if os.path.exists(file):
        os.remove(file)

def appendFileToFile(file, append):
    fin = open(append, "r")
    data2 = fin.read()
    fin.close()
    fout = open(file, "a")
    fout.write(data2)
    fout.close()

    # Remove the append file
    if os.path.exists(append):
        os.remove(append)

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

def getCurrentTemp():
    result = runBashCommand("rocm-smi")
    line = result.split('\n')[5]
    # Find temperature
    temp = float(re.search(r"..\..c", line).group().replace("c", ""))

    return temp

def appendCurrentTemp(file):
    """Writes the current GPU Temperature to an file

    Args:
        file - a path to a filepath.
    Returns:
        temp - current GPU temperature.
    """
    result = runBashCommand("rocm-smi")
    line = result.split('\n')[5]
    # Find temperature
    temp = re.search(r"..\..c", line).group()

    with open(file, "a+") as benchFile:
        benchFile.write("Temperature: " + str(temp.replace("c", " c")) + "\n")

    return temp

def appendStringToFile(string, file):
    assert type(string) is str 
    with open(file, "a+") as benchFile:
        benchFile.write(string + "\n")

def benchmarkCommand(benchmark, folder, levelCore, levelMem, typeEx, explore):
    """Constructs the bash command with default naming scheme
    Args:
        benchmark: benchmark command.
        folder: folder where benchmark is.

    Returns:
        command - command generated
        file - output file name and path
    """
    global CoreFreq
    global CoreVoltage
    global MemoryFreq
    global MemoryVoltage
    name = benchmark.copy()
    name[0] = str(benchmark[0]).replace(str(folder) + '/', '')
    name = '_'.join(name)

    benchmark = ' '.join(benchmark)

    command = str(benchmark)
    file = str(folder) + "/Results/" + name + "-" + str(typeEx) + "-" + str(
        explore) + "-" + "Core-" + str(levelCore) + "-" + str(
            CoreFreq[levelCore]) + '-' + str(
                CoreVoltage[levelCore]) + "-Memory-" + str(
                    levelMem) + '-' + str(MemoryFreq[levelMem]) + '-' + str(
                        MemoryVoltage[levelMem]) + ".txt"

    return command, file


def runBashCommandOutputToFile(bashCommand, filePath, execution):
    """Runs a bash command and outputs the process stdout and stderr to file

    Args:
        bashCommand: Command to be run
        filePath: Path of the output file

    Returns:
        The resulting process
    """
    print("Running %s" % (bashCommand))
    with open(filePath, "a+") as output_file:
        output_file.write("\n")
        output_file.write("#################################################\n")
        output_file.write("Execution: " + str(execution) + " .\n")
    appendCurrentTemp(filePath)
    
    seconds = 150
    try:
        if "3.6" in sys.version:
            with open(filePath, "a+") as output_file:
                process = subprocess.run(bashCommand.split(),
                                         stdout=output_file,
                                         stderr=output_file,
                                         stdin=subprocess.PIPE,
                                         check=True,
                                         timeout=seconds)

                # Write GPU Temp to end of output file
                output_file.write("Status: Success .\n#################################################\n\n")
            print("\tSuccess", filePath)

            return True, process.stdout.decode("utf-8") 
        else:
            with open(filePath, "a+") as output_file:
                process = subprocess.run(bashCommand.split(),
                                         stdout=output_file,
                                         stderr=output_file,
                                         check=True,
                                         text=True,
                                         timeout=seconds)

                # Write GPU Temp to end of output file
                output_file.write("Status: Success .\n#################################################\n\n")
            print("\tSuccess", filePath)

            return True, process.stdout, process.returncode
    except subprocess.CalledProcessError as e:
        print("\n\tERROR: Execution %s .\n" % (str(e.returncode)))
        with open(filePath, "a+") as output_file:
            output_file.write("Status: ERROR - Execution %s .\n" % (str(e.returncode)))
            output_file.write("#################################################\n\n")

        return False, None, e.returncode
    except subprocess.TimeoutExpired as t:
        print("\n\tERROR: Timeout\n")
        with open(filePath, "a+") as output_file:
            output_file.write("Status: ERROR - Timeout .\n")
            output_file.write("#################################################\n\n")

    return False, None, 0



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

def setCoreAndMemPerformanceLevel(coreLevel, memoryLevel):
    assert coreLevel in list(range(0, 8)), "Core Performance Level betwen 0 and 7."
    assert memoryLevel in list(range(0, 4)), "Core Performance Level betwen 0 and 3."
    result = runDVFSscript("-P " + str(level))
    if "ERROR" in result:
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

    return CoreFreq.index(core), MemoryFreq.index(mem)

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

def getOverdrive():
    result = runBashCommand("rocm-smi")
    line = result.split('\n')[5]
    line = line[20:]
    # Find powercap value
    powercap = re.search(r"...\..W", line).group()
    powercap = float(powercap.replace("W", ""))

    return powercap

def setOverDrive():
    if getOverdrive() != float(320):
        # Enables Overdrive
        result = runBashCommand("rocm-smi --autorespond y --setoverdrive 20")
        if "Successfully" not in result:
            print("Not able to reset GPU")
            exit()

        result = runBashCommand("rocm-smi --autorespond y --setpoweroverdrive 320")
        if "Successfully" not in result:
            print("Not able to reset GPU")
            exit()

def preDVFSconfig():
    setOverDrive()

    # Set Core performance level to the one to be tested
    if setPerformanceLevel("core", 0) == False:
        print(" Not able to select core level.")
        return False
    # Set Memory performance level to the highest
    if setPerformanceLevel("mem", 0) == False:
        print(" Not able to select memory level.")
        return False

    return True

def exportDVFStable():
    global CoreFreq
    global CoreVoltage
    global MemoryFreq
    global MemoryVoltage

    file = open("currentDVFS.txt","w") 
    for freq, volt in zip(CoreFreq, CoreVoltage):
        file.write(str(freq) + "," + str(volt) + "\n") 
    for freq, volt in zip(MemoryFreq, MemoryVoltage):
        file.write(str(freq) + "," + str(volt) + "\n")      
    file.close() 

def setFan(value):
    assert type(value) is int
    assert value < 256
    assert value >= 0
    
    # Set GPU fan to 100%
    result = runBashCommand("rocm-smi --setfan " + str(value))
    if "Successfully" not in result:
        print("Not able to set fan")
        exit()

# Parser to collect user given arguments
parser = argparse.ArgumentParser(
    prog="exploreDVFS",
    description='Run Benchmark and Perform DVFS Exploration')
group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('-b',
                    '--benchmark',
                    metavar='path',
                    type=str,
                    help="Name of the benchmark",
                    required=True,
                    nargs='+')
parser.add_argument('-e', '--experiment',
                    const=1,
                    default=0,
                    action='store_const',
                    help="The output of the benchmark determines if the run was sucessful")
parser.add_argument('-r', '--reset',
                    const=1,
                    default=0,
                    action='store_const',
                    help="Reset the DVFS table between runs")
parser.add_argument('-c',
                    const=1,
                    default=0,
                    action='store_const',
                    help="Performs exploration on GPU Core")
parser.add_argument('-m',
                    const=1,
                    default=0,
                    action='store_const',
                    help="Performs exploration on GPU Memory")
group.add_argument('-v',
                   const=1,
                   default=0,
                   help="Performs exploration of voltage",
                   action='store_const')
group.add_argument('-f',
                   const=1,
                   default=0,
                   help="Performs exploration of frequency",
                   action='store_const')
parser.add_argument('-lc',
                    '--levelscore',
                    help="Performance Levels to be explored on Core",
                    nargs='+',
                    type=int,
                    choices=range(0, 8))
parser.add_argument('-lm',
                    '--levelsmemory',
                    help="Performance Levels to be explored on Memory",
                    nargs='+',
                    type=int,
                    choices=range(0, 4))
parser.add_argument('-t',
                    '--tries',
                    default=10,
                    help="Number of times to perform the benchmark",
                    type=int,
                    choices=range(0, 51))
parser.add_argument('-s',
                    '--step',
                    default=10,
                    help="Step between runs",
                    type=int,
                    choices=range(5, 51))
parser.add_argument('--config',
                    metavar='path',
                    type=str,
                    help="Config file",
                    nargs='+')
args = parser.parse_args()

if args.levelscore == None:
    args.levelscore = list(range(0, 8))

if args.levelsmemory == None:
    args.levelsmemory = list(range(0, 4))

if args.c == 1:
    print("Exploration of Core -", end=" ")
if args.m == 1:
    print("Exploration of Memory -", end=" ")
if args.v == 1:
    print("volt", end=" ")
if args.f == 1:
    print("frequency")
print(args.benchmark, end=" ")
print(args.config)
print()


# Reset GPU Power Table
result = runBashCommand("rocm-smi -r")
if "Successfully" not in result:
    print("Not able to reset GPU")
    exit()

# Disable DVFS
result = runBashCommand("rocm-smi --setperflevel manual")
if "Successfully" not in result:
    print("Not able to set manual performance level")
    exit()

# Set GPU fan to 100%
setFan(255)

# Enables Overdrive
result = runBashCommand("rocm-smi --autorespond y --setoverdrive 20")
if "Successfully" not in result:
    print("Not able to reset GPU")
    exit()

result = runBashCommand("rocm-smi --autorespond y --setpoweroverdrive 320")
if "Successfully" not in result:
    print("Not able to reset GPU")
    exit()


if args.config is not None:
    Core = []
    Mem = []
    # Parse config file
    config_file = open(" ".join(args.config), "r")
    while True:
        line = config_file.readline()
        if not line:
            break

        # Check the type of content of the output file
        parseType = re.search(r"(.*?)\=", line)
        if parseType is not None:
            # Get the information written on the name of the file
            if parseType.group(0) == "defaultCore=":
                for i in range(0, 8):
                    outputLine = config_file.readline()
                    if not outputLine:
                        break
                    Core.append([int(n) for n in outputLine.replace('\n', '').split(",")])
            # Get the expected output lines
            if parseType.group(0) == "defaultMemory=":
                for i in range(0, 4):
                    outputLine = config_file.readline()
                    if not outputLine:
                        break
                    Mem.append([int(n) for n in outputLine.replace('\n', '').split(",")])
    for pair in Core:
        a, b = pair
        CoreFreq.append(str(a))
        CoreVoltage.append(str(b))

    for pair in Mem:
        a, b = pair
        MemoryFreq.append(str(a))
        MemoryVoltage.append(str(b))

    # Set Core performance level to the one to be tested
    if setPerformanceLevel("core", 0) == False:
        print(" Not able to select core level.")
        exit()
    # Set Memory performance level to the highest
    if setPerformanceLevel("mem", 0) == False:
        print(" Not able to select memory level.")
        exit()

    if editAllPerformanceLevels() == False:
        print("not able to update table for current run")
        exit()
else:
    # Get the current power table
    process = runBashCommand("rocm-smi -S")
    i = 0
    for line in process.split('\n'):
        if i > 4 and i < 13:
            CoreFreq.append(line.replace(':', '').split()[2].replace("Mhz", ''))
            CoreVoltage.append(line.replace(':', '').split()[3].replace("mV", ''))
        if i > 13 and i < 18:
            MemoryFreq.append(line.replace(':', '').split()[2].replace("Mhz", ''))
            MemoryVoltage.append(
                line.replace(':', '').split()[3].replace("mV", ''))
        i = i + 1

# Export current DVFS Table
exportDVFStable()

print(args.benchmark)
print(args.benchmark[0][:args.benchmark[0].rfind('/')])
# Checks if the benchmark exists and create a Results folder
folder = str(args.benchmark[0][:args.benchmark[0].rfind('/')])
if not os.path.isdir(folder) or not os.path.isfile(args.benchmark[0]):
    print("Benchmark doesn't exist!")
    exit()
Path(folder + "/Results").mkdir(parents=True, exist_ok=True)

# Exploration of Core and Memory
if args.c == 1 and args.m == 1:
    # Activates intended performance levels
    workingCore = [0] * 8
    lastCore = [0] * 8
    for i in args.levelscore:
        workingCore[i] = 1
    workingMemory = [0] * 4
    lastMemory = [0] * 8
    for i in args.levelsmemory:
        workingMemory[i] = 1

    while 1:
        # Run the benchmark for the proposed levels
        for levelsCore in args.levelscore:
            for levelsMemory in args.levelsmemory:
                # Run the benchmark multiple times
                i = 0
                while i < args.tries:
                    print("Try number: ", i)
                    # Places PowerPlay Table to current values
                    if args.reset == 1:
                        if preDVFSconfig() == False:
                            print("not able to update table for current run")
                            continue

                        if editAllPerformanceLevels() == False:
                            print("not able to update table for current run")
                            continue
                    # Set Core performance level to the one to be tested
                    if setPerformanceLevel("core", int(levelsCore)) == False:
                        print(" Not able to select core level.")
                        continue
                    # Set Memory performance level to the highest
                    if setPerformanceLevel("mem", int(levelsMemory)) == False:
                        print(" Not able to select memory level.")
                        continue

                    # Run warm up DVFS program
                    runBashCommand("./warmup")

                    # Get current DVFS settings - to make sure it was correctly applyed
                    cur = currentPerfLevel()
                    if cur != (int(levelsCore), int(levelsMemory)):
                        print(" Selected Performance Levels don't match current ones. %s != (%d, %d)" % (cur, int(levelsCore), int(levelsMemory)))
                        continue

                    # Command to be launch
                    if args.v == 1:
                        commandBenchmark, fileBenchmark = benchmarkCommand(
                            args.benchmark, folder, levelsCore, levelsMemory, "CoreExploration",
                            "Voltage")
                    else:
                        commandBenchmark, fileBenchmark = benchmarkCommand(
                            args.benchmark, folder, levelsCore, levelsMemory, "CoreExploration",
                            "Frequency")

                    # Checks if the intended voltage is correctly applied to the GPU
                    result, volt = currentVoltageIsRespected(CoreVoltage[int(levelsCore)], MemoryVoltage[int(levelsMemory)])
                    print(result, volt)
                    if result == False:
                        print("Current voltage is %d != Core: %d | Memory: %d" % (int(volt), int(CoreVoltage[int(levelsCore)]), int(MemoryVoltage[int(levelsMemory)])))
                        continue

                    # Run the benchmark
                    result, output = runBashCommandOutputToFile(commandBenchmark, fileBenchmark, i)
                    i += 1

        if args.v == 1:
            # Undervolt Core by 10mV
            for editLevel in range(0,8):
                if int(CoreVoltage[editLevel]) - args.step >= 810:
                    CoreVoltage[editLevel] = int(CoreVoltage[editLevel]) - args.step
                else:
                    CoreVoltage[editLevel] = 800 + editLevel * 2
                    if lastCore[editLevel] == 1:
                        workingCore[editLevel] = 0
                    lastCore[editLevel] = 1

            # Undervolt Memory by 10mV
            for editLevel in range(0,4):
                if int(MemoryVoltage[editLevel]) - args.step >= 810:
                    MemoryVoltage[editLevel] = int(MemoryVoltage[editLevel]) - args.step
                else:
                    MemoryVoltage[editLevel] = 800 + editLevel * 2
                    if lastMemory[editLevel] == 1:
                        workingMemory[editLevel] = 0
                    lastMemory[editLevel] = 1
        else:
            # Overclock all levels Core by 10Hz
            CoreFreq = [int(volt) + args.step for volt in CoreFreq]

        # Apply new Power Table Settings
        for levels in range(0, 8):
            if editPerformanceLevel("core", levels, CoreFreq[levels], CoreVoltage[levels]) == False:
                print("Failed to update level %d" % (levels))
                workingworkingCore[levels] = 0
        # Apply new Power Table Settings
        for levels in range(0, 4):
            if editPerformanceLevel("mem", levels, MemoryFreq[levels], MemoryVoltage[levels]) == False:
                print("Failed to update level %d" % (levels))
                workingMemory[levels] = 0

        # Export current DVFS Table
        exportDVFStable()

# Exploration of Core
elif args.c == 1:
    # Activates intended performance levels
    working = [0] * 8
    last = [0] * 8
    for i in args.levelscore:
        working[i] = 1

    while 1 in working:
        # Run the benchmark for the proposed levels
        for levels in args.levelscore:
            # Check if level is still giving valid results
            if working[levels] == 0:
                continue
            # Run the benchmark multiple times
            i = 0
            failedExec = 0
            failedPerfLevel = 0
            failedVoltage = 0
            failedInside = 0
            setFan(255)
            curTemp = getCurrentTemp()
            print("Current Temperature: ", curTemp)
            while curTemp > float(40):
                # Set GPU fan to 100%
                setFan(255)
                time.sleep(2)
                curTemp = getCurrentTemp()
                print("Current Temperature: ", curTemp)
            while i < args.tries:
                print("Try number: ", i)
                # Set GPU fan to 100%
                setFan(255)
                curTemp = getCurrentTemp()
                print("Current Temperature: ", curTemp)
                while curTemp > float(45):
                    # Set GPU fan to 100%
                    setFan(255)
                    time.sleep(2)
                    curTemp = getCurrentTemp()
                    print("Current Temperature: ", curTemp)

                # Places PowerPlay Table to current values
                if args.reset == 1:
                    if preDVFSconfig() == False:
                        print("not able to update table for current run")
                        continue

                    if editAllPerformanceLevels() == False:
                        print("not able to update table for current run")
                        continue
                # Set Core performance level to the one to be tested
                if setPerformanceLevel("core", int(levels)) == False:
                    print(" Not able to select core level.")
                    continue
                # Set Memory performance level to the highest
                if setPerformanceLevel("mem", 3) == False:
                    print(" Not able to select memory level.")
                    continue

                # Run warm up DVFS program
                runBashCommand("./warmup")

                # Guarantee that a reset on the GPU doesn't reset the overdrive settings
                setOverDrive()

                # Get current DVFS settings - to make sure it was correctly applyed
                cur = currentPerfLevel()
                if cur != (int(levels), 3):
                    print(" Selected Performance Levels don't match current ones. %s != (%d, 3)" % (cur, int(levels)))
                    failedPerfLevel += 1
                    if failedPerfLevel > 2:
                        appendStringToFile("failedPerfLevel", fileBenchmark)
                        break
                    continue

                # Command to be launch
                if args.v == 1:
                    commandBenchmark, fileBenchmark = benchmarkCommand(args.benchmark, folder, levels, 3, "CoreExploration", "Voltage")

                    # Checks if the intended voltage is correctly applied to the GPU
                    result, volt = currentVoltageIsRespected(CoreVoltage[int(levels)], MemoryVoltage[3])
                    print(result, volt)
                    if result == False:
                        print("Current voltage is %d != Core: %d | Memory: %d" % (int(volt), int(CoreVoltage[int(levels)]), int(MemoryVoltage[3])))
                        failedVoltage += 1
                        if failedVoltage > 2:
                            appendStringToFile("failedVoltage", fileBenchmark)
                            break
                        continue
                else:
                    commandBenchmark, fileBenchmark = benchmarkCommand(args.benchmark, folder, levels, 3, "CoreExploration", "Frequency")

                # Run the benchmark
                if args.experiment == 1:
                    # Remove temp file to guarantee that no trash is in it
                    removeFile("output.txt")
                    # Run the benchmark
                    result, output, returncode = runBashCommandOutputToFile(commandBenchmark, "output.txt", i)
                    print("Return code: ", returncode)
                    if returncode == 255:
                        failedPerfLevel += 1
                        if failedPerfLevel > 2:
                            break
                        continue
                    appendFileToFile(fileBenchmark, "output.txt")
                    if returncode != 0:
                        failedInside += 1
                        if failedInside > 0:
                            appendStringToFile("failedInside", fileBenchmark)
                            break
                else:
                    result, output, returncode = runBashCommandOutputToFile(commandBenchmark, fileBenchmark, i)

                if result == False:
                    failedExec += 1
                    if failedExec > 0:
                        working[levels] = 0
                        appendStringToFile("failedExec", fileBenchmark)
                        break;
                
                i += 1
                failedPerfLevel = 0
                failedVoltage = 0

        if args.v == 1:
            # Undervolt Core by 10mV
            for editLevel in range(0,8):
                if int(CoreVoltage[editLevel]) - args.step >= 810:
                    CoreVoltage[editLevel] = int(CoreVoltage[editLevel]) - args.step
                else:
                    CoreVoltage[editLevel] = 800 + editLevel * 2
                    if last[editLevel] == 1:
                        working[editLevel] = 0
                    last[editLevel] = 1
            if int(CoreVoltage[7]) < 900:
                exit()
            # if int(CoreFreq[7]) > 1600 and int(CoreVoltage[7]) < 1050:
            #     exit()
            # if int(CoreFreq[7]) > 1500 and int(CoreVoltage[7]) < 1000:
            #     exit()
            # if int(CoreFreq[7]) > 1349 and int(CoreVoltage[7]) < 950:
                exit()
        else:
            # Overclock all levels Core by 10Hz
            CoreFreq = [int(volt) + args.step for volt in CoreFreq]

        # Apply new Power Table Settings
        for levels in range(0, 8):
            if editPerformanceLevel("core", levels, CoreFreq[levels], CoreVoltage[levels]) == False:
                print("Failed to update level %d" % (levels))
                working[levels] = 0

        # Export current DVFS Table
        exportDVFStable()

# Exploration of Memory
elif args.m == 1:
    # Activates intended performance levels
    working = [0] * 4
    last = [0] * 4
    for i in args.levelsmemory:
        working[i] = 1

    while 1 in working:
        # Run the benchmark for the proposed levels
        for levels in args.levelsmemory:
            # Check if level is still giving valid results
            if working[levels] == 0:
                continue
            # Run the benchmark multiple times
            i = 0
            failedExec = 0
            failedPerfLevel = 0
            failedVoltage = 0
            failedInside = 0
            while i < args.tries:
                print("Try number: ", i)
                # Set GPU fan to 100%
                setFan(255)
                curTemp = getCurrentTemp()
                print("Current Temperature: ", curTemp)
                while curTemp > float(45):
                    # Set GPU fan to 100%
                    setFan(255)
                    time.sleep(2)
                    curTemp = getCurrentTemp()
                    print("Current Temperature: ", curTemp)

                # Places PowerPlay Table to current values
                if args.reset == 1:
                    if preDVFSconfig() == False:
                        print("not able to update table for current run")
                        continue
                    if editAllPerformanceLevels() == False:
                        print("not able to update table for current run")
                        continue

                # Set Core performance level to the one to be tested
                if setPerformanceLevel("core", 7) == False:
                    print(" Not able to select core level.")
                    continue
                # Set Memory performance level to the highest
                if setPerformanceLevel("mem", int(levels)) == False:
                    print(" Not able to select memory level.")
                    continue

                # Run warm up DVFS program
                runBashCommand("./warmup")

                # Guarantee that a reset on the GPU doesn't reset the overdrive settings
                setOverDrive()

                # Get current DVFS settings - to make sure it was correctly applyed
                cur = currentPerfLevel()
                if cur != (7, int(levels)):
                    print(" Selected Performance Levels don't match current ones. %s != (7, %d)" % (cur, int(levels)))
                    failedPerfLevel += 1
                    if failedPerfLevel > 2:
                        appendStringToFile("failedPerfLevel", fileBenchmark)
                        break
                    continue

                # Command to be launch
                if args.v == 1:
                    commandBenchmark, fileBenchmark = benchmarkCommand(args.benchmark, folder, 7, levels, "MemoryExploration", "Voltage")
                    # Checks if the intended voltage is correctly applied to the GPU
                    result, volt = currentVoltageIsRespected(CoreVoltage[7], MemoryVoltage[int(levels)])
                    print(result, volt)
                    if result == False:
                        print("Current voltage is %d != Core: %d | Memory: %d" % (int(volt), int(CoreVoltage[7]), int(MemoryVoltage[int(levels)])))
                        failedVoltage += 1
                        if failedVoltage > 2:                            
                            appendStringToFile("failedVoltage", fileBenchmark)
                            break
                        continue
                else:
                    commandBenchmark, fileBenchmark = benchmarkCommand(args.benchmark, folder, 7, levels, "MemoryExploration", "Frequency")

                if args.experiment == 1:
                    # Remove temp file to guarantee that no trash is in it
                    removeFile("output.txt")
                    # Run the benchmark
                    result, output, returncode = runBashCommandOutputToFile(commandBenchmark, "output.txt", i)
                    print("Return code: ", returncode)
                    # If return code is 255 it means that the DVFS config wasn't correctly applied
                    if returncode == 255:
                        failedPerfLevel += 1
                        if failedPerfLevel > 2:
                            break
                        continue
                    appendFileToFile(fileBenchmark, "output.txt")
                    if returncode != 0:
                        failedInside += 1
                        if failedInside > 2:
                            appendStringToFile("failedInside", fileBenchmark)
                            break
                else:
                    # Run the benchmark
                    result, output, returncode = runBashCommandOutputToFile(commandBenchmark, fileBenchmark, i)
                
                if result == False:
                    failedExec += 1
                    if failedExec > 2:
                        working[levels] = 0
                        appendStringToFile("failedExec", fileBenchmark)
                        break;
                        
                i += 1
                failedPerfLevel = 0
                failedVoltage = 0

        if args.v == 1:
            # Undervolt Memory by 10mV
            for editLevel in range(0,4):
                if int(MemoryVoltage[editLevel]) - args.step >= 810:
                    MemoryVoltage[editLevel] = int(MemoryVoltage[editLevel]) - args.step
                else:
                    MemoryVoltage[editLevel] = 800 + editLevel * 2
                    if last[editLevel] == 1:
                        working[editLevel] = 0
                    last[editLevel] = 1
        else:
            # Overclock Memory by 10Hz
            MemoryFreq = [int(freq) + args.step for freq in MemoryFreq]

        # Apply new Power Table Settings
        for levels in range(0, 4):
            if editPerformanceLevel("mem", levels, MemoryFreq[levels], MemoryVoltage[levels]) == False:
                print("Failed to update level %d" % (levels))
                working[levels] = 0

        # Export current DVFS Table
        exportDVFStable()

else:
    print("No indication of exploration given [v:voltage, f:frequency]")

# GPU fan automatic
result = runBashCommand("rocm-smi --resetfans")
if "Successfully" not in result:
    print("Not able to set fan")
    exit()
