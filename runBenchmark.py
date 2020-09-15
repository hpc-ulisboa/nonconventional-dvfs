import argparse
import subprocess
from pathlib import Path
import os
import re
import sys
import json

CoreFreq = []
CoreVoltage = []
MemoryFreq = []
MemoryVoltage = []


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
    output_file = open(filePath, 'a')
    output_file.write("\n")
    output_file.write("#################################################\n")
    output_file.write("Execution: " + str(execution) + "\n")
    output_file.write("#################################################\n")
    output_file.write("\n")
    if "3.6" in sys.version:
        process = subprocess.run(bashCommand.split(),
                                 stdout=output_file,
                                 stderr=output_file,
                                 stdin=subprocess.PIPE,
                                 check=True)
        return process.stdout.decode("utf-8") 
    else:
        process = subprocess.run(bashCommand.split(),
                                 stdout=output_file,
                                 stderr=output_file,
                                 check=True,
                                 text=True)
    return process.stdout


def runBashCommand(bashCommand):
    """Runs a bash command

    Args:
        bashCommand: Command to be run

    Returns:
        The resulting process
    """
    print("Running %s" % (bashCommand))
    if "3.6" in sys.version:
        process = subprocess.run(bashCommand.split(),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 stdin=subprocess.PIPE,
                                 check=True)
        return process.stdout.decode("utf-8") 
    else:
        process = subprocess.run(bashCommand.split(),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             check=True,
                             text=True)
    return process.stdout

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

    print("File ", file)
    with open(file, "a+") as benchFile:
        benchFile.write("Temperature: " + str(temp.replace("c", " c")) + "\n")

    return temp

def currentVoltageIsRespected(currentVolt):
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

    if abs(int(volt) - int(currentVolt)) <= 5:
        return True, volt

    return False, volt

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
result = runBashCommand("rocm-smi --setfan 255")
if "Successfully" not in result:
    print("Not able to set fan")
    exit()

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

# Checks if the benchmark exists and create a Results folder
folder = str(args.benchmark[0][:args.benchmark[0].rfind('/')])
if not os.path.isdir(folder) or not os.path.isfile(args.benchmark[0]):
    print("Benchmark doesn't exist!")
    exit()
Path(folder + "/Results").mkdir(parents=True, exist_ok=True)

# Exploration of Core and Memory
if args.c == 1 and args.m == 1:
    # Activates intended performance levels
    working_core = [0] * 8
    for i in args.levelscore:
        working_core[i] = [0] * 4
        for j in args.levelscore:
            working_core[i][j] = 1

    while 1 in working:
        # Run the benchmark for the proposed levels
        for levels in args.levelscore:
            # Check if level is still giving valid results
            if working[levels] == 0:
                continue
            # Set Core performance level to the one to be tested
            if setPerformanceLevel("core", int(levels)) == False:
                working[levels] = 0
                continue
            # Set Memory performance level to the highest
            if setPerformanceLevel("mem", 3) == False:
                working[levels] = 0
                continue
            # Get current DVFS settings - to make sure it was correctly applyed
            if currentPerfLevel() != (int(levels), 3):
                working[levels] = 0
                continue
            # Run the benchmark multiple times
            for i in range(0, args.tries):
                # Command to be launch
                if args.v == 1:
                    commandBenchmark, fileBenchmark = benchmarkCommand(
                        args.benchmark, folder, levels, 3, "CoreExploration",
                        "Voltage")
                else:
                    commandBenchmark, fileBenchmark = benchmarkCommand(
                        args.benchmark, folder, levels, 3, "CoreExploration",
                        "Frequency")

                # Run the benchmark
                runBashCommandOutputToFile(commandBenchmark, fileBenchmark, i)

                # Write GPU Temp to end of output file
                appendCurrentTemp(fileBenchmark)

        if args.v == 1:
            # Undervolt Core by 10mV
            CoreVoltage = [int(volt) - 10 for volt in CoreVoltage]
        else:
            # Overclock all levels Core by 10Hz
            CoreFreq = [int(volt) + 10 for volt in CoreFreq]

        # Apply new Power Table Settings
        for levels in range(0, 8):
            if setPerformanceLevel("core", levels) == False:
                working[levels] = 0

# Exploration of Core
elif args.c == 1:
    # Activates intended performance levels
    working = [0] * 8
    last = [0] * 8
    for i in args.levelscore:
        working[i] = 1

    
    # Run the benchmark for the proposed levels
    for levels in args.levelscore:
        # Run the benchmark multiple times
        for i in range(0, args.tries):
            # Places PowerPlay Table to current values
            if args.reset == 1:
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

            runBashCommand("./warm_up_GPU 2")

            # Get current DVFS settings - to make sure it was correctly applyed
            cur = currentPerfLevel()
            if cur != (int(levels), 3):
                print(" Selected Performance Levels don't match current ones. %s != (%d, 3)" % (cur, int(levels)))
                continue

            # Command to be launch
            if args.v == 1:
                commandBenchmark, fileBenchmark = benchmarkCommand(
                    args.benchmark, folder, levels, 3, "CoreExploration",
                    "Voltage")

                # Checks if the intended voltage is correctly applied to the GPU
                result, volt = currentVoltageIsRespected(CoreVoltage[int(levels)])
                print(result, volt)
                if result == False:
                    print("Current voltage is %d != %d" % (int(volt), int(levels)))
                    continue
            else:
                commandBenchmark, fileBenchmark = benchmarkCommand(
                    args.benchmark, folder, levels, 3, "CoreExploration",
                    "Frequency")

            # Run the benchmark
            runBashCommandOutputToFile(commandBenchmark, fileBenchmark, i)

            # Write GPU Temp to end of output file
            appendCurrentTemp(fileBenchmark)

    

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
            for i in range(0, args.tries):
                # Places PowerPlay Table to current values
                if args.reset == 1:
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
                # Get current DVFS settings - to make sure it was correctly applyed
                cur = currentPerfLevel()
                if cur != (7, int(levels)):
                    print(" Selected Performance Levels don't match current ones. %s != (7, %d)" % (cur, int(levels)))
                    continue

                # Command to be launch
                if args.v == 1:
                    commandBenchmark, fileBenchmark = benchmarkCommand(
                        args.benchmark, folder, 7, levels, "MemoryExploration",
                        "Voltage")
                else:
                    commandBenchmark, fileBenchmark = benchmarkCommand(
                        args.benchmark, folder, 7, levels, "MemoryExploration",
                        "Frequency")

                # Run the benchmark
                runBashCommandOutputToFile(commandBenchmark, fileBenchmark, i)
                # Write GPU Temp to end of output file
                appendCurrentTemp(fileBenchmark)

        if args.v == 1:
            # Undervolt Memory by 10mV
            for editLevel in range(0,4):
                if int(MemoryVoltage[editLevel]) - 10 > 810:
                    MemoryVoltage[editLevel] = int(MemoryVoltage[editLevel]) - 10
                else:
                    MemoryVoltage[editLevel] = 800 + editLevel * 2
                    if last[editLevel] == 1:
                        working[editLevel] = 0
                    last[editLevel] = 1
        else:
            # Overclock Memory by 10Hz
            MemoryFreq = [int(freq) + 10 for freq in MemoryFreq]

        # Apply new Power Table Settings
        for levels in range(0, 4):
            if editPerformanceLevel("mem", levels, MemoryFreq[levels], MemoryVoltage[levels]) == False:
                print("Failed to update level %d" % (levels))
                working[levels] = 0

else:
    print("No indication of exploration given [v:voltage, f:frequency]")

# GPU fan automatic
result = runBashCommand("rocm-smi --resetfans")
if "Successfully" not in result:
    print("Not able to set fan")
    exit()
