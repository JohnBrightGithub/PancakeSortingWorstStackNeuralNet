import os
#cleans duplicate rows out of files
#duplicates are added because it is generally not worth parsing an entire file to check if the row exists before writing
printDiff = True

def setPrintDiff(newDiff):
    global printDiff
    printDiff = newDiff

def getPrintDiff():
    return printDiff

def cleanDuplicatesFromFile(filePath):
    lines_seen = {} # holds lines already seen
    lines_removed = {}
    outfile = open(filePath, "r")
    for line in outfile:
        line = line.rstrip()
        if line in lines_seen:
            lines_removed[line] = 1
        else:
            lines_seen[line]=1
    outfile = open(filePath, "w")
    outfile.truncate(0)
    for line in lines_seen:
        if not line==None:
            outfile.write(line + "\n")
    outfile.close()
    if(printDiff):
        if(len(lines_removed)!=0):
            print("lines removed: ")
            for line in lines_removed:
                print(line)

def cleanDuplicatesFolder(Folder):
    directory = os.fsencode(Folder)
    for file in os.listdir(directory):
        fileName = Folder+"/"+os.fsdecode(file)
        if(os.path.isdir(fileName)):
            cleanDuplicatesFolder(fileName)
        else:
            print("cleaning duplicates from file: ", fileName)
            cleanDuplicatesFromFile(fileName)

def cleanDuplicatesResults():
    cleanDuplicatesFolder("Results")

def cleanDuplicatesTraining():
    cleanDuplicatesFolder("Training")

def cleanDuplicatesAll():
    cleanDuplicatesResults()
    cleanDuplicatesTraining()
