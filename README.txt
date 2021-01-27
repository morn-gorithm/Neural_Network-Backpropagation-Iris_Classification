Program takes in 3 runtime arguments in the run command:
	java -jar IrisClassifier.jar args[0] args[1] args[2]

Order of runtime arguments :
	args[0] = Name of training data file ("train.data")
	args[1] = Name of test data file ("iris.data")
	args[2] = Name of parameter file ("parameter.txt")

Command to run program using given files :
	java -jar IrisClassifier.jar test.data iris.data parameter.txt

Parameter File values:
	[0] : learning Rate
	[1] : Batch Size (Number of inputs within a single batch)
	[2] : Maximum Number of epochs before convergence

Program results and output will be put in a file named "output.txt",
Program will also print some basic information to the console.