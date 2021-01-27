import java.util.Random;
import java.util.Scanner;
import java.io.File;
import java.io.IOException;
import java.io.FileWriter;

public class Main {

  public static void main(String[] args) throws Exception {
    //order of cmd line args:
    //[0] : training file
    //[1] : test file
    //[2] : learningParams (learning Rate)
    //the number of neurons in each layer
    int[] layerSize = {
      4,
      5,
      3
    };
    //zValues will store the weighted input into a neuron (the input value to a neurons activation function)
    double[][] zValues = {
      new double[4],
      new double[5],
      new double[3]
    };
    //aValues will store the activated values for all neurons
    double[][] aValues = {
      new double[4],
      new double[5],
      new double[3]
    };
    //deltaWeightValues will store the gradient descent values for each weight. These values will change with each training input
    double[][][] deltaWeightValues = {
      new double[4][5],
      new double[5][3]
    };
    //errorValues will store the error values for each node in the hidden and output layer
    double[][] errorValues = {
      new double[4],
      new double[5],
      new double[3]
    };
    //stores the delta in bias values per layer
    double[][] deltaBiasValues = {
      new double[5],
      new double[3]
    };
    //init zValues to 0
    for (int i = 0; i < 3; i++) {
      for (int k = 0; k < layerSize[i]; k++) {
        zValues[i][k] = 0;
      }
    }
    System.out.println("---------------------------------Neural Network Variable Details---------------------------------");
    //Each neuron in the hidden and output layer will have its own bias added to its weighted input
    double[][] biasArr = {
      new double[5],
      new double[3]
    };
    //weightArr is a 3d array that holds the weight between input->hidden and hidden -> output
    double[][][] weightArr = {
      new double[4][5],
      new double[5][3]
    };
    double learningRate = 0.001;
    int maxEpochs = 15000;
    int numBatches = 10;
    int batchSize = 15;
    String paramFile = args[2];
    File learnFile = new File(paramFile);
    Scanner ls = new Scanner(learnFile);
    Scanner learnR = null;
    for (int i = 0; i < 3; i++) {
      if (ls.hasNextLine()) {
        learnR = new Scanner(ls.nextLine());
      }
      learnR.useDelimiter("\\s");
      while (learnR.hasNext()) {
        if (i == 0) {
          learningRate = learnR.nextDouble();
          System.out.println("Learing Rate : " + learningRate);
        }
        else if (i == 1) {
          batchSize = learnR.nextInt();
          System.out.println("Batch Size : " + batchSize);
        }
        else {
          maxEpochs = learnR.nextInt();
          System.out.println("max Epochs: " + maxEpochs);
        }

      }
    }
    //Reading in Test Data
    // File testFile = new File(args[0]);
    //--------------------------INPUT ARRAY AND EXPECTED OUTPUT ARRAY INITIALIZATION ------------------------------------
    File trainFile = new File(args[0]);
    Scanner sc = new Scanner(trainFile);
    int inputCount = 0;
    while (sc.hasNextLine()) {
      if (! ("".equals(sc.nextLine().trim()))) {
        inputCount++;
      }
    }
    //inputArr will hold the 4 variables for all input instances
    double[][] inputArr = new double[inputCount][4];
    //inputExpectedOutput will hold the expected classification of the index- corresponding input instance 0-Setosa 1- versicolor 2- virginia
    int[] inputExpectedOut = new int[inputCount];
    sc = new Scanner(trainFile);
    //linescanner will insert individual input instance values into relevant arrays
    Scanner lineScanner = null;
    //varCount will be used to track what variable is to be inserted next. If it gets to 2 the next variable is a flower string expected output
    int varCount = 0;
    //flowerName will be used to convert a flower to an int representation
    String flowerName = "";
    for (int i = 0; i < inputCount; i++) {
      if (sc.hasNextLine()) {
        lineScanner = new Scanner(sc.nextLine());
      }
      lineScanner.useDelimiter(",");
      while (lineScanner.hasNext()) {
        if (varCount < 4) {
          if (lineScanner.hasNextDouble()) {
            inputArr[i][varCount] = (lineScanner.nextDouble());
          }
        }
        else {
          flowerName = lineScanner.next();
          if ((flowerName.toLowerCase()).contains("setosa")) {
            inputExpectedOut[i] = 0;
          }
          else if ((flowerName.toLowerCase()).contains("versicolor")) {
            inputExpectedOut[i] = 1;
          }
          else if ((flowerName.toLowerCase()).contains("virginica")) {
            inputExpectedOut[i] = 2;
          }
        }
        varCount++;
      }
      varCount = 0;
    } //inputArray and expected Output Array are initialized with corresponding indices
    numBatches = inputCount / batchSize;
    double bmin = -1.0;
    double bmax = 1.0;
    double wmin = -1.0;
    double wmax = 1.0;
    //Hidden Layer Biases initialized to random double values
    for (int i = 0; i < layerSize[1]; i++) {
      biasArr[0][i] = randomGenerator(bmin, bmax);
    }
    //output layer biases initialized
    for (int i = 0; i < layerSize[2]; i++) {
      biasArr[1][i] = randomGenerator(bmin, bmax);
    }
    //initialized input layer to hidden layer weights
    for (int i = 0; i < layerSize[0]; i++) {
      for (int j = 0; j < layerSize[1]; j++) {
        weightArr[0][i][j] = randomGenerator(wmin, wmax);
      }
    }
    //initializing hidden layer to output layer weights
    for (int i = 0; i < layerSize[1]; i++) {
      for (int j = 0; j < layerSize[2]; j++) {
        weightArr[1][i][j] = randomGenerator(wmin, wmax);
      }
    }
    //input counter used to track how many inputs are remaining
    //----------------------Outer Loop For Running epochs and generating batches until convergence--------------------
    boolean converged = false;
    int epochCounter = 0;
    double minCost = 0.0;
    double[] cost = new double[maxEpochs];
    int convergCounter = 0;
    int correctTrain = 0;
    System.out.println("---------------------------------LEARNING USING BACKPROPAGATION---------------------------------");
    while (!converged && epochCounter < maxEpochs && convergCounter < 15) {
      double[] errorValue = new double[inputCount];
      double[][][] batches = new double[numBatches][batchSize][4];
      double[][] batchesActualOutput = new double[numBatches][batchSize];
      int[] vstd = new int[inputCount];
      for (int i = 0; i < inputCount; i++) {
        vstd[i] = -1;
      }
      int addedInput = 0;
      for (int i = 0; i < numBatches; i++) {
        for (int j = 0; j < batchSize; j++) {
          int nextIndexInputBatch = randomInt(0, inputCount - 1);
          nextIndexInputBatch = nextTrainIndex(vstd, nextIndexInputBatch);
          for (int k = 0; k < 4; k++) {
            batches[i][j][k] = inputArr[nextIndexInputBatch][k];
          }
          batchesActualOutput[i][j] = inputExpectedOut[nextIndexInputBatch];
          vstd[addedInput] = nextIndexInputBatch;
          addedInput++;
        }
      }
      //batches now contain all inputs split randomly into sub-batches
      for (int a = 0; a < numBatches; a++) {
        for (int b = 0; b < batchSize; b++) {
          //setting initial z values for input neurons
          zValues[0] = normalizeInput(batches[a][b]);
          double neuronVal = 0;
          double[] inputHolder = new double[4];
          for (int m = 0; m < 4; m++) {
            inputHolder[m] = zValues[0][m];
            //input values are activated and stored in aValues array
            aValues[0][m] = zValues[0][m];
          }
          //calculate ZValues for hidden layer
          for (int i = 0; i < layerSize[1]; i++) {
            for (int j = 0; j < layerSize[0]; j++) {
              neuronVal += (aValues[0][j] * weightArr[0][j][i]);
            }
            neuronVal += biasArr[0][i];
            zValues[1][i] = neuronVal;
            neuronVal = 0;
          }
          neuronVal = 0;
          //calculate A Values for hidden layer neurons
          for (int j = 0; j < layerSize[1]; j++) {
            aValues[1][j] = (relu(zValues[1][j]));
          }
          // calculate z values for output layer neurons
          for (int i = 0; i < layerSize[2]; i++) {
            for (int j = 0; j < layerSize[1]; j++) {
              neuronVal += (aValues[1][j]) * (weightArr[1][j][i]);
            }
            neuronVal += biasArr[1][i];
            //the z values for output layer are stored
            zValues[2][i] = neuronVal;
            neuronVal = 0;
          }
          //calculate A values for output layer neurons
          for (int i = 0; i < layerSize[2]; i++) {
            aValues[2][i] = softMax(zValues[2], i);
          }
          //All Z and A Values should be set and therefore feedforward is complete
          double error = crossEntropyLoss(aValues[2], batchesActualOutput[a][b]);
          cost[epochCounter] += error;
          //-----------------------------------------------------------------------------BACKPROP------------------------------------
          //Calculate the Hidden -> Outer weight adjustments for this instance
          double[] expectedProbability = oneHotEncode(batchesActualOutput[a][b]);
          for (int i = 0; i < layerSize[2]; i++) {
            for (int j = 0; j < layerSize[1]; j++) {
              //tempDelta will store the derivative of cost with respect to weight /_\W
              double tempDelta = 0.0;
              tempDelta = outputWeightDelta(aValues, j, i, batchesActualOutput[a][b]);
              deltaWeightValues[1][j][i] += tempDelta;
            }
          }
          //Calculate Hidden -> Output bias adjustments for this instance
          for (int i = 0; i < layerSize[2]; i++) {
            deltaBiasValues[1][i] += biasDelta(outputNodeError(i, aValues[2], batchesActualOutput[a][b]));
          }
          //Calculate Input -> Hidden weight adjustments
          for (int i = 0; i < layerSize[1]; i++) {
            for (int j = 0; j < layerSize[0]; j++) {
              deltaWeightValues[0][j][i] += hiddenWeightDelta(weightArr[1], aValues, zValues[1], j, i, batchesActualOutput[a][b]);
            }
          }
          //Calculate Input -> Hidden Bias adjustments
          for (int i = 0; i < layerSize[1]; i++) {
            deltaBiasValues[0][i] += biasDelta(hiddenNodeError(weightArr[1], aValues[2], zValues[1], i, batchesActualOutput[a][b]));
          }
        } // next input instance in the batch after this curly brace
      } //new batch after this curly brace
      //------------WEIGHT AND BIAS CORRECTION--------------------
      //now we adjust the weights and biases according to results obtained from current batch
      //input -> hidden weights
      for (int i = 0; i < layerSize[1]; i++) {
        for (int k = 0; k < layerSize[0]; k++) {
          weightArr[0][k][i] = updateWeight(weightArr[0][k][i], deltaWeightValues[0][k][i], learningRate, batchSize);
        }
      }
      //HIDDEN->OUTPUT weights
      for (int i = 0; i < layerSize[2]; i++) {
        for (int k = 0; k < layerSize[1]; k++) {
          weightArr[1][k][i] = updateWeight(weightArr[1][k][i], deltaWeightValues[1][k][i], learningRate, batchSize);
        }
      }
      //hidden bias adjustments
      for (int i = 0; i < layerSize[1]; i++) {
        biasArr[0][i] = updateBias(biasArr[0][i], deltaBiasValues[0][i], learningRate, batchSize);
      }
      //average output bias adjustments
      for (int i = 0; i < layerSize[2]; i++) {
        biasArr[1][i] = updateBias(biasArr[1][i], deltaBiasValues[1][i], learningRate, batchSize);
      }
      double[] nmlzd = new double[4];
      correctTrain = 0;
      for (int i = 0; i < inputCount; i++) {
        nmlzd = normalizeInput(inputArr[i]);
        double[] prediction = new double[layerSize[2]];
        for (int a = 0; a < layerSize[2]; a++) {
          double hidden = 0;
          for (int b = 0; b < layerSize[1]; b++) {
            double input = 0;
            for (int c = 0; c < layerSize[0]; c++) {
              input += nmlzd[c] * weightArr[0][c][b];
            }
            hidden = input + biasArr[0][b];
            hidden = relu(hidden);
            prediction[a] += hidden * weightArr[1][b][a];
          }
          prediction[a] += biasArr[1][a];
        }
        double[] probPredict = new double[layerSize[2]];
        double targ = 0.0;
        for (int a = 0; a < layerSize[2]; a++) {
          probPredict[a] = softMax(prediction, a);
        }
        if (inputExpectedOut[i] == 0) {
          cost[epochCounter] += crossEntropyLoss(probPredict, inputExpectedOut[i]);
        }
        else if (inputExpectedOut[i] == 1) {
          cost[epochCounter] += crossEntropyLoss(probPredict, inputExpectedOut[i]);
        }
        else {
          cost[epochCounter] += crossEntropyLoss(probPredict, inputExpectedOut[i]);
        }
        double highestProb = 0;
        int probHighInd = 0;
        for (int k = 0; k < 3; k++) {
          if (probPredict[k] >= highestProb) {
            probHighInd = k;
            highestProb = probPredict[k];
          }
        }
        if (inputExpectedOut[i] == probHighInd) {
          correctTrain++;
        }
      }
      cost[epochCounter] = cost[epochCounter] / inputCount;
      if (epochCounter == 0) {
        minCost = cost[epochCounter];
      }
      else {
        if (cost[epochCounter] < minCost) {
          if (cost[epochCounter] - minCost > -0.000002) {
            convergCounter++;
          }
          minCost = cost[epochCounter];
        }
        else {
          convergCounter++;
        }
      }
      for (int i = 0; i < layerSize[1]; i++) {
        for (int k = 0; k < layerSize[0]; k++) {
          deltaWeightValues[0][k][i] = 0;
        }
        deltaBiasValues[0][i] = 0;
      }
      for (int i = 0; i < layerSize[2]; i++) {
        for (int k = 0; k < layerSize[1]; k++) {
          deltaWeightValues[1][k][i] = 0;
        }
        deltaBiasValues[1][i] = 0;
      }
      epochCounter++;
    } //end of epoch while
    System.out.println("---------------------------------TESTING--------------------------------------------------------");
    File testFile = new File(args[1]);
    Scanner tSc = new Scanner(testFile);
    int testCount = 0;
    while (tSc.hasNextLine()) {
      if (! ("".equals(tSc.nextLine().trim()))) {
        testCount++;
      }
    }
    double[][] testArr = new double[testCount][4];
    int[] testExpectedOutput = new int[testCount];
    tSc = new Scanner(testFile);
    Scanner testLineScanner = null;
    int testVarCount = 0;
    String testFlowerName = "";
    for (int i = 0; i < testCount; i++) {
      if (tSc.hasNextLine()) {
        testLineScanner = new Scanner(tSc.nextLine());
      }
      testLineScanner.useDelimiter(",");
      while (testLineScanner.hasNext()) {
        if (testVarCount < 4) {
          if (testLineScanner.hasNextDouble()) {
            testArr[i][testVarCount] = (testLineScanner.nextDouble());
          }
        }
        else {
          testFlowerName = testLineScanner.next();
          if ((testFlowerName.toLowerCase()).contains("setosa")) {
            testExpectedOutput[i] = 0;
          }
          else if ((testFlowerName.toLowerCase()).contains("versicolor")) {
            testExpectedOutput[i] = 1;
          }
          else if ((testFlowerName.toLowerCase()).contains("virginica")) {
            testExpectedOutput[i] = 2;
          }
        }
        testVarCount++;
      }
      testVarCount = 0;
    }
    //will iterate through each input for testing
    int numCorrect = 0;
    for (int t = 0; t < testCount; t++) {
      zValues[0] = normalizeInput(testArr[t]);
      double neuronVal = 0;
      double[] inputHolder = new double[4];
      for (int m = 0; m < 4; m++) {
        inputHolder[m] = zValues[0][m];
        //input values are activated and stored in aValues array
        aValues[0][m] = zValues[0][m];
      }
      //Input Layer => Hidden Layer Neruon Val pre Activation
      for (int i = 0; i < layerSize[1]; i++) {
        for (int j = 0; j < layerSize[0]; j++) {
          neuronVal += (aValues[0][j] * weightArr[0][j][i]);
        }
        neuronVal += biasArr[0][i];
        //the hidden layer zValues are stored
        zValues[1][i] = neuronVal;
        neuronVal = 0;
      }
      neuronVal = 0;
      //-Apply the activation function on the hidden layer neruon values for each neuron
      //-Sum the total of the activated values * weight
      //-add the hidden layer bias and assign to output neurons
      for (int i = 0; i < layerSize[2]; i++) {
        for (int j = 0; j < layerSize[1]; j++) {
          //each hidden layer neuron is activated and the value is stored
          aValues[1][j] = (relu(zValues[1][j]));
          neuronVal += (aValues[1][j]) * (weightArr[1][j][i]);
        }
        neuronVal += biasArr[1][i];
        //the z values for output layer are stored
        zValues[2][i] = neuronVal;
        neuronVal = 0;
      }
      //-Apply the activation function on the output layer neruon
      for (int i = 0; i < layerSize[2]; i++) {
        aValues[2][i] = softMax(zValues[2], i);
      }
      double highestProbability = 0;
      int probabilityHighest = 0;
      for (int i = 0; i < 3; i++) {
        if (aValues[2][i] >= highestProbability) {
          probabilityHighest = i;
          highestProbability = aValues[2][i];
        }
      }
      System.out.println("Program Predicts Class: " + probabilityHighest + " | Actual Class : " + testExpectedOutput[t] + " with Confidence " + aValues[2][probabilityHighest]);
      if (probabilityHighest == testExpectedOutput[t]) {
        numCorrect++;
      }
    }
    //now aValues at last layer will be the predicted output values of all the test inputs
    //-----------------------------Writing To Output File -----------------------------
    try {
      File output = new File("output.txt");
      if (output.createNewFile()) {
        //file created
      }
      FileWriter outWriter = new FileWriter("output.txt");
      outWriter.write("Weight Matrix  \n");
      outWriter.write("Input Layer [i] to Hidden Layer [j] Weights \n");
      for (int h = 0; h < 5; h++) {
        for (int i = 0; i < 4; i++) {
          outWriter.write("W[" + i + "][" + h + "] : " + weightArr[0][i][h] + "\t");
        }
        outWriter.write("\n");
      }
      outWriter.write("\n");
      outWriter.write("Hidden Layer [j] to Output Layer [k] Weights \n");
      for (int k = 0; k < 3; k++) {
        for (int j = 0; j < 5; j++) {
          outWriter.write("W[" + j + "][" + k + "] : " + weightArr[1][j][k] + "\t");
        }
        outWriter.write("\n");
      }
      outWriter.write("\n");
      outWriter.write("Biases To Hidden Layer Neuron [n]: \n");
      for (int i = 0; i < 5; i++) {
        outWriter.write("B[" + i + "] : " + biasArr[0][i] + "\n");
      }
      outWriter.write("\n");
      outWriter.write("Biases To Output Layer Neuron [n]: \n");
      for (int i = 0; i < 3; i++) {
        outWriter.write("B[" + i + "] : " + biasArr[1][i] + "\n");
      }
      outWriter.write("\n");
      outWriter.write("Number of Epochs Taken To Converge : " + epochCounter);
      outWriter.write("\n");
      outWriter.write("\n");
      outWriter.write("Accuracy for Training Set : \nTotal Correct: " + correctTrain);
      outWriter.write("\nTotal Training Instances : " + inputCount);
      double corr = correctTrain;
      double accuracy = corr / inputCount;
      accuracy = accuracy * 100;
      outWriter.write("\nAccuracy : " + (accuracy) + "%");
      outWriter.write("\n\n");
      outWriter.write("Accuracy for Testing Set : \nTotal Correct: " + numCorrect);
      outWriter.write("\nTotal Testing Instances : " + testCount);
      corr = numCorrect;
      accuracy = corr / testCount;
      accuracy = accuracy * 100;
      outWriter.write("\nAccuracy : " + (accuracy) + "%");
      outWriter.close();
    }
    catch(IOException e) {
      System.out.println("Error Writing to Output File");
      e.printStackTrace();
    }

  } //end of Main
  //---------------------------------HELPER FUNCTIONS-----------------------------------
  public static double[] feedForward(double[] inputArray, double[][][] weights, double[][] biasArray, double learnRate) {
    double[] nmlzd = normalizeInput(inputArray);
    double[] prediction = new double[3];
    for (int a = 0; a < 3; a++) {
      double hidden = 0;
      for (int b = 0; b < 5; b++) {
        double input = 0;
        for (int c = 0; c < 3; c++) {
          input += nmlzd[c] * weights[0][c][b];
        }
        hidden = input + biasArray[0][b];
        hidden = relu(hidden);
        prediction[a] += hidden * weights[1][b][a];
      }
      prediction[a] += biasArray[1][a];
    }
    double[] probPredict = new double[3];
    double targ = 0.0;
    for (int a = 0; a < 3; a++) {
      probPredict[a] = softMax(prediction, a);
    }
    return probPredict;
  }
  public static double[] normalizeInput(double[] input) {
    double mean = 0.0;
    for (int i = 0; i < 4; i++) {
      mean += input[i];
    }
    mean = mean / 4;
    double variance = 0.0;
    for (int i = 0; i < 4; i++) {
      variance += Math.pow((input[i] - mean), 2);
    }
    variance = variance / 4;
    double stdDev = Math.sqrt(variance);
    double[] normalized = new double[4];
    for (int i = 0; i < 4; i++) {
      normalized[i] = (input[i] - mean) / stdDev;
    }
    return normalized;

  }
  //this function will return the error value of a specific node in the hidden layer ( layer =0) or the output layer (layer =1)
  public static double outputNodeError(int index, double[] avals, double actual) {
    double error = 0.0;
    double[] actualValue = oneHotEncode(actual);
    error = avals[index] - actualValue[index];
    return error;
  }
  public static double hiddenNodeError(double[][] weights, double[] avals, double[] zvals, int index, double actual) {
    double error = 0.0;
    for (int i = 0; i < 3; i++) {
      error += outputNodeError(i, avals, actual) * weights[index][i];
    }
    error = error * derivRelu(zvals[index]);
    return error;
  }
  public static double outputWeightDelta(double[][] avals, int inputIndex, int outputIndex, double expected) {
    //error of neuron that weight is going to * activation of input node
    return outputNodeError(outputIndex, avals[2], expected) * avals[1][inputIndex];
  }
  public static double hiddenWeightDelta(double[][] weights, double[][] avals, double[] zvals, int inputIndex, int outputIndex, double actual) {
    double error = 0;
    error = hiddenNodeError(weights, avals[2], zvals, outputIndex, actual) * avals[0][inputIndex];
    return error;
  }
  public static double biasDelta(double error) {
    return error;
  }
  public static double updateWeight(double currentWeight, double totalDeltaW, double learnRate, int batchSize) {
    double averageDelta = totalDeltaW / batchSize;
    return currentWeight - (learnRate * averageDelta);
  }
  public static double updateBias(double currentBias, double totalDeltaB, double learnRate, int batchSize) {
    double averageDelta = totalDeltaB / batchSize;
    return currentBias - (learnRate * averageDelta);
  }

  //random number generator
  private static Random gen = new Random();
  public static double randomGenerator(double rMin, double rMax) {
    double randomVal = rMin + (rMax - rMin) * gen.nextDouble();
    return randomVal;
  }
  private static Random genInt = new Random();
  public static int randomInt(int min, int max) {
    int difference = max - min;
    difference++;
    return (int)(difference * Math.random()) + min;
  }
  public static boolean contains(int[] arr, int i) {
    boolean exists = false;
    for (int j = 0; j < arr.length; j++) {
      if (arr[j] == i) {
        exists = true;
      }
    }
    return exists;
  }
  //nextTrainIndex will take in a potential index for the training set, if that index has already been used then it returns the nearest available index that hasnt been tested
  public static int nextTrainIndex(int[] arr, int i) {
    int down = 0;
    int up = 0;
    if (contains(arr, i)) {
      for (int k = 0; k < arr.length; k++) {
        //check below
        if ((i - k) >= 0) {
          if (!contains(arr, (i - k))) {
            return i - k;
          }
        }
        if ((i + k) < arr.length) {
          if (!contains(arr, (i + k))) {
            return i + k;
          }
        }
      }
    }
    return i;
  }
  //relu is the activation function for all non output layers
  public static double relu(double n) {
    if (n < 0) {
      return 0.0;
    }
    else return n;
  }
  public static double derivRelu(double n) {
    if (n > 0) {
      return 1.00;
    }
    else {
      return 0.0;
    }
  }
  //softmax is the activation function for output layer and calculates a probability distribution (so out of 1 what is the probability that the
  public static double softMax(double[] n, int ind) {
    double temp = 0.0;
    for (int i = 0; i < 3; i++) {
      temp += Math.exp(n[i]);
    }
    return (Math.exp(n[ind]) / temp);
  }
  //the softmax Deriv in terms of an output layers Z value
  public static double softMaxDeriv(double[] z, int ind) {
    double temp = 0.0;
    double nume = 0.0;
    for (int i = 0; i < 3; i++) {
      temp += Math.exp(z[i]);
      if (i != ind) {
        nume += Math.exp(z[i]);
      }
    }
    temp = Math.pow(temp, 2); //denominator
    nume = nume * z[ind];
    return (nume) / temp;
  }
  //cross entropy loss will use the activation value of output layer to calculate an error Value for output layer
  //softmax calculates probability distribution and therefore cannot be compared to 0,1,2 values of the flowers
  public static double crossEntropyLoss(double[] pred, double targ) {
    double[] probTarg = oneHotEncode(targ);
    double temp = 0;
    int targetIndex = 0;
    for (int i = 0; i < 3; i++) {
      if (probTarg[i] == 1.0 || probTarg[i] == 1) {
        targetIndex = i;
      }
    }
    for (int i = 0; i < 3; i++) {
      temp = temp + (probTarg[i] * (Math.log(pred[i])));
    }
    return - temp;
  }
  //one hot encode will take in the expected result and turn it into a probability distribution between the 3 possible results.
  //i.o.w the function will return an array with the target result being = 1 and the other results = 0
  //so if the expected result is a Iris-setosa with class value 0 then the matrix return will be [1,0,0]
  public static double[] oneHotEncode(double expectedResult) {
    double[] oneHE = {
      0,
      0,
      0
    };
    if (expectedResult == 1.0 || expectedResult == 1) {
      oneHE[1] = 1.0;
    }
    else if (expectedResult == 2.0 || expectedResult == 2) {
      oneHE[2] = 1.0;
    }
    else {
      oneHE[0] = 1.0;
    }
    return oneHE;
  }
}