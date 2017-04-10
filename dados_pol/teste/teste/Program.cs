using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Globalization;
using System.Threading;

//for PCA
using Accord.Math;
using Accord.Statistics;

namespace teste
{
    class Program
    {
        static void Main(string[] args)
        {

            //***************************************language sintax********************************
            //*************************************************************************************
            CultureInfo culture;
            if (Thread.CurrentThread.CurrentCulture.Name != "en-US")
                culture = CultureInfo.CreateSpecificCulture("en-US");
            else
                culture = CultureInfo.CreateSpecificCulture("en-US");

            Thread.CurrentThread.CurrentCulture = culture;
            Thread.CurrentThread.CurrentUICulture = culture;

            //this was just to confirm that each sample has 65 values (index 0 to 63 =64 values are features, index 64 is the class)
            //double[] a = new double[] { 0.20055, 0.37951, 0.39641, 2.0472, 32.351, 0.38825, 0.24976, 1.3305, 1.1389, 0.50494, 0.24976, 0.6598, 0.1666, 0.24976, 497.42, 0.73378, 2.6349, 0.24976, 0.14942, 43.37, 1.2479, 0.21402, 0.11998, 0.47706, 0.50494, 0.60411, 1.4582, 1.7615, 5.9443, 0.11788, 0.14942, 94.14, 3.8772, 0.56393, 0.21402, 1.741, 593.27, 0.50591, 0.12804, 0.66295, 0.051402, 0.12804, 114.42, 71.05, 1.0097, 1.5225, 49.394, 0.1853, 0.11085, 2.042, 0.37854, 0.25792, 2.2437, 2.248, 348690, 0.12196, 0.39718, 0.87804, 0.001924, 8.416, 5.1372, 82.658, 4.4158, 7.4277, 0 };

            //****iris dataset****
            /*
            Console.WriteLine("\nData is the famous Iris flower set.");
            Console.WriteLine("Predict species from sepal length, width, petal length, width");
            Console.WriteLine("Iris setosa = 0 0 1, versicolor = 0 1 0, virginica = 1 0 0 \n");
            Console.WriteLine("Raw data resembles:");
            Console.WriteLine(" 5.1, 3.5, 1.4, 0.2, Iris setosa");
            Console.WriteLine(" 7.0, 3.2, 4.7, 1.4, Iris versicolor");
            Console.WriteLine(" 6.3, 3.3, 6.0, 2.5, Iris virginica");
            Console.WriteLine(" ......\n");
            
            //double[][] allData = new double[150][];
            */

            ///*

            //*********************declare random numbers***************
            //******************************************************
            double randN = 0;
            int randN_int = 0;


            //**************************CHOOSE DATA*********************************
            //************************************************************************
            //goal: read the file

            //change here <=======================
            //string dataFile_orig = "..\\..\\IrisData.txt";  //iris dataset
            //string dataFile_orig = "..\\..\\1year.txt";   //1st year data
            string dataFile_orig = "..\\..\\allyears.txt";  //data from all years 

            //**************************Proprieties of data**********************
            //**************************************************************************
            //goal: declare up-front the data characteristics

            //decide if data has missing values
            //change here <=====================
            bool missing_values = true;
            if (dataFile_orig == "..\\..\\allyears.txt" || dataFile_orig == "..\\..\\1year.txt")
                missing_values = true;
            else if (dataFile_orig == "..\\..\\IrisData.txt")
                missing_values = false;

            //change the character used for missing values //----------------------------------------------should be replaced by number detection?
            string missing_char = "?";

            //define number of features in the data
            //change here <=======================
            int num_features = 0; 
            if (dataFile_orig == "..\\..\\allyears.txt" || dataFile_orig == "..\\..\\1year.txt") //then it has 1 class
                num_features = 64;
            else if (dataFile_orig == "..\\..\\IrisData.txt")
                num_features = 6;
            int numCols = num_features + 1; //number of columns per sample
            int numRows = 0; //number of lines of file

            // decide if dataset is imbalanced
            //change here <=======================
            bool is_balanced = true;
            if (dataFile_orig == "..\\..\\allyears.txt" || dataFile_orig == "..\\..\\1year.txt")
                is_balanced = false;
            else if (dataFile_orig == "..\\..\\IrisData.txt")
                is_balanced = true;

            //decide if there is data to encode
            //change here <=======================
            bool encode_input = false;
            bool encode_output = true;

            //dclare working datafile name
            string dataFile = null;
            double[][] allData_array = null;
            string allData_string = null;

            //***********************MISSING VALUES*************************************
            //**************************************************************************
            //goal: replace missing values with some substitute value

            //SKIP
            //change here <=======================
            bool already_did_this = true;
            //bool already_did_this = false;
            if (!already_did_this)
            {
            if (missing_values)
                {
                double substitute = 0;

                //change here <=====================
                //decide method
                string subs_method = "random";
                //string subs_method = "mean";

                if (subs_method=="random")          //random number between -1 and 1
                    substitute = RandomNumberBetween(-1, 1);
                else if (subs_method == "mean")
                {

                }

            //read file to a string
            dataFile = dataFile_orig;
            string text = File.ReadAllText(dataFile);

            //replace '?' by a random number
            text = text.Replace(missing_char, substitute.ToString());

             //change name of file
             dataFile = dataFile_orig + "_no_missing.txt";

            //save string to a file
            File.WriteAllText(dataFile, text);
                }
            }
            else //skipped
            {
                if (missing_values)
                    dataFile = dataFile_orig + "_no_missing.txt";
            }

            //*************************IMBALANCED DATA**********************
            //***************************************************************************
            //goal: create a balanced dataset

            //SKIP
            //change here <=======================
            already_did_this = true;
            //already_did_this = false;
            if (!already_did_this)
            {
                //transfer the data from filename into a array(array)
                allData_array=TextToArray(dataFile, numCols);

                if (!is_balanced)
            { 
            //convert data array to a list
            var allData_list = new List<double[]>(allData_array);
            //Console.WriteLine("\nFirst 6 rows of the data set:");
            //ShowMatrix(allData_list.ToArray(), 0, 6, 1, true);

            //count the number of positive (=="faliram") cases. It's the ones with class value 1
            numRows = allData_list.Count;
            int num_positive = 0;
            for (int i=0; i<numRows; i++)
            {
                if (allData_list[i][num_features] == 1)
                    num_positive++;
            }
            int num_negative = numRows - num_positive; //negative samples are those that ("nao faliram")

            //keep only the same amount of positively and negatively classified samples, delete the rest
            int deleted = 0;
            int to_delete = num_negative - num_positive;
            while (deleted < to_delete)
            {
                randN = RandomNumberBetween(0, numRows); //choose 1 line/sample randomly
                randN_int = (int) randN;
                if (allData_list[randN_int][num_features] == 0) //if it's a negative sample, delete it
                { 
                    allData_list.RemoveAt(randN_int);
                    numRows--;
                    to_delete--;
            }
            }

            //convert back from list to array
            allData_array = allData_list.ToArray();

            //save in a file
            allData_string = ArrayToString(allData_array);
            
            //change name of file
            dataFile = dataFile_orig + "_balanced.txt";

            //save to a file
            File.WriteAllText(dataFile, allData_string);
                }
 }
            else //skipped
            {
                if(!is_balanced)
                dataFile = dataFile_orig + "_balanced.txt";
            }
            //**********************************ENCODE DATA***************************
            //**********************************************************************
            //goal: encode the data

            //SKIP
            //change here <=====================
            already_did_this = false;
            if (!already_did_this) {

                //update data array
                allData_array = TextToArray(dataFile, numCols);

                //change name of file
                string dataFile_encoded = dataFile_orig+"_encoded.txt"; //encoded file

            //decide input variables to encode
            if (encode_input)
            {
                //change here <=====================
                int[] input_index = new int[] { };
                foreach(int index in input_index)
                    EncodeFile(dataFile, dataFile_encoded, index-1, "effects"); //EncodeFile is 0-based

                //change name of file
                dataFile = dataFile_encoded;
                    
                    //update data array
                    allData_array = TextToArray(dataFile, numCols);
                }


            //decide output variables to encode
            if (encode_output)
            {
                //change here <=====================
                int[] output_index = new int[] {num_features+1}; //EncodeFile is 0 based
                foreach (int index in output_index)
                    EncodeFile(dataFile, dataFile_encoded, index - 1, "dummy"); //EncodeFile is 0-based

                // change name of file
                dataFile = dataFile_encoded;

                    //update data array - not necessary
                    //allData_array = TextToArray(dataFile, numCols);
                }
                //data array to string
                //allData_string = ArrayToString(allData_array);
                //save string to a file
                //File.WriteAllText(dataFile, allData_string);

                //choose "effects" if variable to encode is an input, "dummy" if it's an output
                //Only one encoding option so far: 1-of-N/Manhattan coding. The representation will have number of variables= number of classes
            }
            else //skipped
            {
                if (encode_input==false || encode_output==false)
                    dataFile=dataFile_orig + "_encoded.txt";
            }
            //***********************************NORMALIZE DATA***************************
            //***************************************************************************
            //goal: use a method to normalize data

            //SKIP
            //change here <=====================
            already_did_this = false;
            if (!already_did_this)
            {
                //update data array
                allData_array = TextToArray(dataFile, numCols);
                //allData_array = LoadData(dataFile, numRows, numCols);

                //change here <=====================
                string normalization = "gauss";
                //string normalization = "min-max";
                //string normalization = "both";

                string min_max_mode = "[-1,1]";
                //string min_max_mode = "[0,1]";

                if (normalization == "gauss")
                {
                    for (int i = 0; i < num_features; i++)    //dont normalize the class values
                    {
                        GaussNormal(allData_array, i); //GaussNormal is 0-based
                    }
                }
                else if (normalization == "min-max")
                {
                    for (int i = 0; i < num_features; i++)    //dont normalize the class values
                    {
                        MinMaxNormal(allData_array, i, min_max_mode);
                    }
                }
                else if (normalization == "both")
                {
                    for (int i = 0; i < num_features; i++)    //dont normalize the class values
                    {
                        GaussNormal(allData_array, i); //GaussNormal is 0-based
                        MinMaxNormal(allData_array, i, min_max_mode);
                    }
                }
                //ShowMatrix(allData_array, 0, 13, 4, true);

                //data array to string
                allData_string = ArrayToString(allData_array);

                //change name of file
                dataFile = dataFile_orig + "_norm.txt";

                //save string to a file
                File.WriteAllText(dataFile, allData_string);
            }
            else //skipped
                    dataFile = dataFile_orig + "_norm.txt";

            //*******************************DIMENSIONALITY REDUCTION************************
            //*********************************************************************************
            /*
            //convert to actual matrix
            int numCols = allData[1].Length;
            int numRows = allData.Length;
            double[,] allData_matrix = new double[numRows,numCols];
            for(int l=0; l<numRows; l++)
            {
                for (int c = 0; c < numCols; c++)
                    allData_matrix[l, c] = allData[l][c];
            }

            // Compute total and average 
            //double[] totals = allData_matrix.Sum();
            //double[] averages = allData_matrix.Mean();

            */


            //*******************************VALIDATION*******************************************
            //**********************************************************************************
            //goal: choose validation method

            //update data array
            allData_array = TextToArray(dataFile, numCols);

            //create a list of neural networks
            List<NeuralNetwork> nn = new List<NeuralNetwork>(); //NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);
            int num_networks = 0;
            //change here <=====================
            bool save_networks = false; //choose to save or delete the networks once used

            string mode = null;
            string error_mode = null;

            //change here <=====================
            //define validation method
            string validation_method = "k-fold cross";
            //string validation_method = "hold-out";

            //change here <=====================
            int numFolds = 10; //number of folds if using k-fold cross validation
            if (validation_method == "hold-out")
            {
                //*****************************************TRAIN and TEST SETS********************
                //**************************************************************************************
                //goal: define train and test sets

                //define the train percentage ratio, relative to the whole set
                //change here <=====================
                double[] train_perc_array = new double[] { 0.7 };
                foreach (var train_perc in train_perc_array)
                {
                    //create the test and train sets from the whole dataset
                    Console.WriteLine("Creating " + train_perc * 100 + "% training and " + (1 - train_perc) * 100 + "% test data matrices");
                    double[][] trainData = null;
                    double[][] testData = null;
                    //change here <=====================
                    int seed = 0; //choose a seed
                    MakeTrainTest(allData_array, seed, out trainData, out testData, train_perc);
                    //Console.WriteLine("\nFirst 3 rows of training data:");
                    //ShowMatrix(trainData, 0, 13, 1, true);
                    //Console.WriteLine("First 3 rows of test data:");
                    //ShowMatrix(testData, 0, 13, 1, true);


                    //*******************************Number of INPUT and OUTPUT cells/nodes*********************************
                    //******************************************************************************************************
                    //goal: define number input, output neurons

                    //define number of inputs
                    int numInput = num_features;  //the bias isn't given explicitly and the output doesnt count
                    //should be 64 for our dataset, 4 for Iris

                    int numOutput = allData_array[1].Length-num_features;  //equals the number of classes if using sofmax activation
                    //should be 2 for our dataset, 3 for Iris

                    //********************************LEARNING PARAMETERS***********************************************
                    //**************************************************************************************************
                    //goal: give the learning parameters

                    //change here <=====================
                    //define "maximum number of epochs/iterations"
                    int[] maxEpochs_array = new int[] { 100 };
                    foreach (var maxEpochs in maxEpochs_array)
                    {
                        //change here <=====================
                        //define "learning rate"
                        double[] learnRate_array = new double[] { 0.3 };
                        foreach (var learnRate in learnRate_array)
                        {
                            //change here <=====================
                            //define "momentum"
                            double[] momentum_array = new double[] { 0.2 };
                            foreach (var momentum in momentum_array)
                            {

                                //*********************************NUMBER OF NODES in hidden layer******************************
                                //*********************************************************************************************
                                //goal: define number of hidden nodes in hidden layer

                                //change here <=====================
                                int[] numHidden_array = new int[] { 2, 10, 20, 30, 44, 45, 50, 60, 70, 80, 90, 100 }; //rule of thumb: (Number of inputs + outputs) x 2/3 

                                foreach (var numHidden in numHidden_array)
                                {
                                    //*********************************TRAINING*********************************************
                                    //***********************************************************************************************
                                    //goal: train the model

                                    Console.WriteLine("\nCreating a " + numInput + "-input, " + numHidden + "-hidden, " + numOutput + "-output neural network");
                                    //Console.Write("Hard-coded tanh function for input-to-hidden and softmax for hidden-to-output activations");
                                    Console.WriteLine("Setting maxEpochs = " + maxEpochs + ", learnRate = " + learnRate + ", momentum = " + momentum);

                                    nn.Add(new NeuralNetwork(numInput, numHidden, numOutput));

                                    //change here <=====================
                                    //define error mode
                                    error_mode = "mean_squared_error";
                                    //error_mode = "mean_cross_entropy_error";

                                    //train
                                    //Console.WriteLine("Beginning training using incremental back-propagation");
                                    nn[num_networks].Train(trainData, maxEpochs, learnRate, momentum, error_mode);
                                    //nn[num_networks].Train(trainData, maxEpochs, learnRate, momentum, trainMode, min_error);
                                    //Console.WriteLine("Training complete");

                                    //get the weights obtained through training
                                    double[] weights = nn[num_networks].GetWeights();

                                    //**********************************Show weights and bias values***************************************
                                    //*****************************************************************************************************
                                    //Console.WriteLine("Final neural network weights and bias values:");
                                    //ShowVector(weights, 10, 3, true);


                                    //****************************************MODEL PERFORMANCE*************************************************
                                    //**************************************************************************************************
                                    //goal: evaluate model performance for train and test set

                                    //------------------------------------------------------------still only using accuracy
                                    mode = "train";
                                    double trainAcc = nn[num_networks].Accuracy(trainData, mode);
                                    Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));

                                    mode = "test";
                                    double testAcc = nn[num_networks].Accuracy(testData, mode);
                                    Console.WriteLine("\nAccuracy on test data = " + testAcc.ToString("F4"));

                                    if(save_networks)
                                        num_networks++;
                                    else
                                        nn.Clear();
                                }
                            }
                        }
                    }
                }
            }
            else if (validation_method == "k-fold cross")
            {
                double[][] trainData = null;
                double[][] testData = null;

                //*******************************Number of INPUT and OUTPUT cells/nodes*********************************
                //******************************************************************************************************
                //goal: define number input, output neurons

                //define number of inputs
                int numInput = num_features;  //the bias isn't given explicitly and the output doesnt count
                                              //should be 64 for our dataset, 4 for Iris

                int numOutput = allData_array[1].Length - num_features;  //equals the number of classes if using sofmax activation
                                                                         //should be 2 for our dataset, 3 for Iris


                                                                         //********************************LEARNING PARAMETERS***********************************************
                                                                         //**************************************************************************************************
                                                                         //goal: give the learning parameters

                //change here <=====================
                //define "maximum number of epochs/iterations"
                int[] maxEpochs_array = new int[] { 100 };
                foreach (var maxEpochs in maxEpochs_array)
                {
                    //change here <=====================
                    //define "learning rate"
                    double[] learnRate_array = new double[] { 0.3 };
                    foreach (var learnRate in learnRate_array)
                    {
                        //change here <=====================
                        //define "momentum"
                        double[] momentum_array = new double[] { 0.2 };
                        foreach (var momentum in momentum_array)
                        {
                            //*********************************NUMBER OF NODES in hidden layer******************************
                            //*********************************************************************************************
                            //goal: define number of hidden nodes in hidden layer

                            //change here <=====================
                            int[] numHidden_array = new int[] { 2, 10, 20, 30, 44, 45, 50, 60, 70, 80, 90, 100 }; //rule of thumb: (Number of inputs + outputs) x 2/3 

                            foreach (var numHidden in numHidden_array)
                            {
                                // mean classification error for a neural network
                                int[] cumWrongCorrect = new int[2]; // cumulative # wrong, # correct
                                 for (int k = 0; k < numFolds; ++k) // for each fold
                                    {
                                     //*******************************TRAIN and TEST DATASETS**************************
                                     //**********************************************************************************
                                     //goal: obtain the test and train datasets for this fold

                                     trainData = GetTrainData(allData_array, numFolds, k); // get the training data for current fold
                                     testData = GetTestData(allData_array, numFolds, k); // the test data for current fold

                                    //*********************************TRAINING*********************************************
                                    //***********************************************************************************************
                                    //goal: train the network

                                    //general info
                                    if (k == 0) //no need to write more than once
                                    { 
                                    Console.WriteLine("\nCreating a " + numInput + "-input, " + numHidden + "-hidden, " + numOutput + "-output neural network\n");
                                    //Console.Write("Hard-coded tanh function for input-to-hidden and softmax for hidden-to-output activations\n");
                                    Console.WriteLine("Setting maxEpochs = " + maxEpochs + ", learnRate = " + learnRate + ", momentum = " + momentum);
                                    }

                                    //create neural network
                                    nn.Add(new NeuralNetwork(numInput, numHidden, numOutput));

                                    //change here <=====================
                                    //choose error mode
                                    error_mode = "mean_squared_error";
                                    //error_mode = "mean_cross_entropy_error";

                                    //train
                                    //Console.WriteLine("Beginning training using incremental back-propagation");
                                    nn[num_networks].Train(trainData, maxEpochs, learnRate, momentum, error_mode);
                                    //Console.WriteLine("Training complete");

                                    //obtain weights
                                    double[] weights = nn[num_networks].GetWeights();

                                    //**********************************Show weights and bias values***************************************
                                    //*****************************************************************************************************
                                    //Console.WriteLine("Final neural network weights and bias values:");
                                    //ShowVector(weights, 10, 3, true);


                                    //**********************************MODEL PERFORMANCE****************************
                                    //***************************************************************************
                                    //goal: evaluate model performance

                                    int[] wrongCorrect = nn[num_networks].WrongCorrect(testData); // get classification results

                                    //simple error measure
                                    double error = (wrongCorrect[0] * 1.0) / (wrongCorrect[0] + wrongCorrect[1]);
                                    cumWrongCorrect[0] += wrongCorrect[0]; // accumulate # wrong
                                    cumWrongCorrect[1] += wrongCorrect[1]; // accumulate # correct
                                    //Console.Write("Fold = " + k + ": wrong = " + wrongCorrect[0] + " correct = " + wrongCorrect[1]);
                                    //Console.WriteLine("    error = " + error.ToString("F4"));

                                    //----------------------------------------------------------------still only using this simple measure

                                    if (save_networks)
                                        num_networks++;
                                    else
                                        nn.Clear();
                                }
                                double mce=(cumWrongCorrect[0] * 1.0) / (cumWrongCorrect[0] + cumWrongCorrect[1]); // mean classification error across folds
                double mca = 1.0 - mce; //mean classification accurracy across folds
                //Console.WriteLine("\nCross-validation complete\n");
                Console.WriteLine("Mean cross-validation classification error = " + mce.ToString("F4"));
                Console.WriteLine("Mean cross-validation classification accuracy = " + mca.ToString("F4"));
                            }
                        }
                    }
                }
            }
            Console.ReadLine(); //stop the cmd line
        } // Main 


        //*********************************Convert text file to array********************************
        //*******************************************************************************************
        //calls LoadData(). After having written into a file, give filename to get the data into the array

        static double[][] TextToArray(string dataFile, int numCols)
        { 
        // read number of lines in the file
        int lineCount = 0;
            using (var reader = File.OpenText(@dataFile))  //read the modified file
            {
                while (reader.ReadLine() != null)
                    lineCount++;
            }
         //create and populate the array
        double[][] allData_array = new double[lineCount][];
        allData_array = LoadData(dataFile, lineCount, numCols); //works for both Iris and our datasets
                                                                //LoadData is 0-based
            return allData_array;
        }

        //********************************Convert array to string*****************************************
        //*******************************************************************************************
        static string ArrayToString(double[][] data_array)
        {
            double numCols = data_array[1].Length;
            double numRows = data_array.Length;
            string data_string = null;
            for (int l = 0; l < numRows; l++)
            {
                for (int c = 0; c < numCols; c++)
                {
                    data_string += data_array[l][c].ToString();
                    if (c < numCols - 1)
                        data_string += ",";
                    else if (c == numCols - 1 && l != numRows - 1)
                        data_string += "\n";
                }
                Console.WriteLine(l); //---------------------------------------------------------flags!
            }
            return data_string;
        }



//****************************************************create a  random number***********************
//**************************************************************************************************
private static readonly Random random = new Random();
        private static double RandomNumberBetween(double minValue, double maxValue)
        {
            var next = random.NextDouble();

            return minValue + (next * (maxValue - minValue));
        }


        //********************************Ways of normalizing and standardizing data***********************************************
        //***********************************************************************************************************************
        static void GaussNormal(double[][] data, int column)
        {
            int j = column; 
            double sum = 0.0;
            for (int i = 0; i < data.Length; ++i)
                sum += data[i][j];
            double mean = sum / data.Length;

            double sumSquares = 0.0;
            for (int i = 0; i < data.Length; ++i)
                sumSquares += (data[i][j] - mean) * (data[i][j] - mean);
            double stdDev = Math.Sqrt(sumSquares / data.Length);

            for (int i = 0; i < data.Length; ++i)
                data[i][j] = (data[i][j] - mean) / stdDev;
        }
        static void MinMaxNormal(double[][] data, int column, String mode)
        {
            int j = column;
            double min = data[0][j];
            double max = data[0][j];
            for (int i = 0; i < data.Length; ++i)
            {
                if (data[i][j] < min)
                    min = data[i][j];
                if (data[i][j] > max)
                    max = data[i][j];
            }
            double range = max - min;
            if (mode == "[0,1]") {
                if (Math.Abs(range) < 0.00000001)
                {
                    for (int i = 0; i < data.Length; ++i)
                        data[i][j] = 0.5;
                    return;
                }

                for (int i = 0; i < data.Length; ++i)
                    data[i][j] = (data[i][j] - min) / range;
            }
            if (mode == "[-1,1]")
            {
                //if (range == 0.0)
                if (Math.Abs(range) < 0.00000001)
                {
                    for (int i = 0; i < data.Length; ++i)
                        data[i][j] = 0;
                    return;
                }
                for (int i = 0; i < data.Length; ++i)
                    data[i][j] = (data[i][j] - (max+min)/2) / (range/2);
            }
        }



        //*************************************************create train and test sets using hold-out method*****************************
        //*****************************************************************************************************************************
        static void MakeTrainTest(double[][] allData, int seed, out double[][] trainData, out double[][] testData, double trainPct)
        {
            Random rnd = new Random(seed);
            int totRows = allData.Length;
            int numTrainRows = (int)(totRows * trainPct); // usually 0.80
            int numTestRows = totRows - numTrainRows;
            trainData = new double[numTrainRows][];
            testData = new double[numTestRows][];

            double[][] copy = new double[allData.Length][]; // Make a reference of data: copy.
            for (int i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];

            for (int i = 0; i < copy.Length; ++i) // scramble row order of copy
            {
                int r = rnd.Next(i, copy.Length); // use Fisher-Yates
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }// Copy first trainRows from copy[][] to trainData[][].
            for (int i = 0; i < numTrainRows; ++i)
                trainData[i] = copy[i];

            for (int i = 0; i < numTestRows; ++i)
                testData[i] = copy[i + numTrainRows];
        }


            //*******************************helper methods to show data****************************
            //**************************************************************************************
        static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i % valsPerRow == 0) Console.WriteLine("");
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }
            if (newLine == true) Console.WriteLine("");
        }


        static void ShowMatrix(double[][] matrix, int starting_point, int numRows, int decimals, bool newLine)//static void ShowMatrix(List<double[]> matrix, int starting_point, int numRows, int decimals, bool newLine)
        {
            for (int i = starting_point; i < numRows; ++i)
            {
                Console.Write(i.ToString().PadLeft(3) + ": ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    if (matrix[i][j] >= 0.0)
                        Console.Write(" ");
                    else Console.Write("-");
                    Console.Write(Math.Abs(matrix[i][j]).ToString("F" + decimals) + " ");
                }
                Console.WriteLine("");
            }
            if (newLine == true) Console.WriteLine("");
        }


        //**********************************Load data from a file and save it on a array**************************
        //**********************************************************************************************************
        static double[][] LoadData(string dataFile, int numRows, int numCols)
        {
            double[][] result = new double[numRows][];
            FileStream ifs = new FileStream(dataFile, FileMode.Open);
            StreamReader sr = new StreamReader(ifs);

            string line = "";
            string[] tokens = null;
            int i = 0;
            while ((line = sr.ReadLine()) != null)
            {
                tokens = line.Split(',');
                result[i] = new double[numCols];
                for (int j = 0; j < numCols; ++j)
                {
                    result[i][j] = double.Parse(tokens[j]);

                }
                ++i;
            }
            sr.Close();
            ifs.Close();
            return result;
        }


        //****************************************Ways of encoding the input and output************************
        //****************************************************************************************************
        static string EffectsEncoding(int index, int N)  //effects encoding
        {
            if (N == 2)
            {
                if (index == 0) return "-1";
                else if (index == 1) return "1";
            }

            int[] values = new int[N - 1];
            if (index == N - 1) // Last item is all -1s. 
            {
                for (int i = 0; i < values.Length; ++i)
                    values[i] = -1;
            }
            else
            {
                values[index] = 1; // 0 values are already there.
            }

            string s = values[0].ToString();
            for (int i = 1; i < values.Length; ++i)
                s += "," + values[i];
            return s;
        }

        static string DummyEncoding(int index, int N)  //dummy encoding
        {
            int[] values = new int[N];
            values[index] = 1;

            string s = values[0].ToString();
            for (int i = 1; i < values.Length; ++i)
                s += "," + values[i];
            return s;
        }

        static void EncodeFile(string originalFile, string encodedFile, int column, string encodingType) //the only method called by user
        {
            // encodingType is "effects" or "dummy" 
            FileStream ifs = new FileStream(originalFile, FileMode.Open);
            StreamReader sr = new StreamReader(ifs);
            string line = "";
            string[] tokens = null;
            Dictionary<string, int> d = new Dictionary<string, int>();
            int itemNum = 0;
            while ((line = sr.ReadLine()) != null)
            {
                tokens = line.Split(','); // assumes items are comma-delimited. 
                if (d.ContainsKey(tokens[column]) == false)
                    d.Add(tokens[column], itemNum++);
            }
            sr.Close();
            ifs.Close();
            int N = d.Count; // number of distinct strings. 

            ifs = new FileStream(originalFile, FileMode.Open);
            sr = new StreamReader(ifs);

            FileStream ofs = new FileStream(encodedFile, FileMode.Create);
            StreamWriter sw = new StreamWriter(ofs);
            string s = null; // result line. 
            while ((line = sr.ReadLine()) != null)
            {
                s = "";
                tokens = line.Split(','); // break apart strings. 
                for (int i = 0; i < tokens.Length; ++i) // Reconstruct. 
                {
                    if (i == column) // encode this string. 
                    {
                        int index = d[tokens[i]]; // 0, 1, 2, or . . . 
                        if (encodingType == "effects")
                            s += EffectsEncoding(index, N) + ",";
                        else if (encodingType == "dummy")
                            s += DummyEncoding(index, N) + ",";
                    }
                    else
                        s += tokens[i] + ",";
                }
                s = s.Remove(s.Length - 1); // remove trailing ','.
                sw.WriteLine(s); // write the string to file. 
            } // while 
            sw.Close(); ofs.Close();
            sr.Close(); ifs.Close();
        }


        //****************************************k-fold cross validation******************************
        //*********************************************************************************************
        static double[][] GetTrainData(double[][] allData, int numFolds, int fold)
        {
            int[][] firstAndLastTest = GetFirstLastTest(allData.Length, numFolds); // first and last index of rows tagged as test data
            int numTrain = allData.Length - (firstAndLastTest[fold][1] - firstAndLastTest[fold][0] + 1); // tot num rows - num test rows
            double[][] result = new double[numTrain][];
            int i = 0; // index into result/test data
            int ia = 0; // index into all data
            while (i < result.Length)
            {
                if (ia < firstAndLastTest[fold][0] || ia > firstAndLastTest[fold][1]) // this is a train row
                {
                    result[i] = allData[ia];
                    ++i;
                }
                ++ia;
            }
            return result;
        }

        static double[][] GetTestData(double[][] allData, int numFolds, int fold)
        {
            // return a reference to test data
            int[][] firstAndLastTest = GetFirstLastTest(allData.Length, numFolds); // first and last index of rows tagged as TEST data
            int numTest = firstAndLastTest[fold][1] - firstAndLastTest[fold][0] + 1;
            double[][] result = new double[numTest][];
            int ia = firstAndLastTest[fold][0]; // index into all data
            for (int i = 0; i < result.Length; ++i)
            {
                result[i] = allData[ia]; // the test data indices are contiguous
                ++ia;
            }
            return result;
        }

        static int[][] GetFirstLastTest(int numDataItems, int numFolds)
        {
            // return[fold][firstIndex][lastIndex] for k-fold cross validation test data
            int interval = numDataItems / numFolds;  // if there are 32 data items and k = num folds = 3, then interval = 32/3 = 10
            int[][] result = new int[numFolds][]; // pair of indices for each fold
            for (int i = 0; i < result.Length; ++i)
                result[i] = new int[2];

            for (int k = 0; k < numFolds; ++k) // 0, 1, 2
            {
                int first = k * interval; // 0, 10, 20
                int last = (k + 1) * interval - 1; // 9, 19, 29 (should be 31)
                result[k][0] = first;
                result[k][1] = last;
            }

            result[numFolds - 1][1] = result[numFolds - 1][1] + numDataItems % numFolds; // 29->31
            return result;
        }

    } // class Program 



    //************************************************the neural network per se*************************************************
    //***************************************************************************************************************************
    public class NeuralNetwork
    {
        private int numInput; // number input nodes
        private int numHidden;
        private int numOutput;

        private double[] inputs;
        private double[][] ihWeights; // input-hidden
        private double[] hBiases;
        private double[] hOutputs;

        private double[][] hoWeights; // hidden-output
        private double[] oBiases;
        private double[] outputs;

        private Random rnd;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            this.inputs = new double[numInput];

            this.ihWeights = MakeMatrix(numInput, numHidden, 0.0);
            this.hBiases = new double[numHidden];
            this.hOutputs = new double[numHidden];

            this.hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
            this.oBiases = new double[numOutput];
            this.outputs = new double[numOutput];

            this.rnd = new Random(0);
            this.InitializeWeights(0); // all weights and biases
        } // ctor

        private static double[][] MakeMatrix(int rows,
          int cols, double v) // helper for ctor, Train
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result[i][j] = v;
            return result;
        }

            public void InitializeWeights(int seed)
        {
            // initialize weights and biases to small random values
            rnd = new Random(seed);
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            double[] initialWeights = new double[numWeights];
            double lo = -0.01;
            double hi = 0.01;
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (hi - lo) * rnd.NextDouble() + lo;
            this.SetWeights(initialWeights);
        }


        public void SetWeights(double[] weights)
        {
            // copy serialized weights and biases in weights[] array
            // to i-h weights, i-h biases, h-o weights, h-o biases
            int numWeights = (numInput * numHidden) +
              (numHidden * numOutput) + numHidden + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array in SetWeights");

            int k = 0; // points into weights param

            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                hBiases[i] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    hoWeights[i][j] = weights[k++];
            for (int i = 0; i < numOutput; ++i)
                oBiases[i] = weights[k++];
        }

        public double[] GetWeights()
        {
            int numWeights = (numInput * numHidden) +
              (numHidden * numOutput) + numHidden + numOutput;
            double[] result = new double[numWeights];
            int k = 0;
            for (int i = 0; i < ihWeights.Length; ++i)
                for (int j = 0; j < ihWeights[0].Length; ++j)
                    result[k++] = ihWeights[i][j];
            for (int i = 0; i < hBiases.Length; ++i)
                result[k++] = hBiases[i];
            for (int i = 0; i < hoWeights.Length; ++i)
                for (int j = 0; j < hoWeights[0].Length; ++j)
                    result[k++] = hoWeights[i][j];
            for (int i = 0; i < oBiases.Length; ++i)
                result[k++] = oBiases[i];
            return result;
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
            double[] oSums = new double[numOutput]; // output nodes sums

            for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
                this.inputs[i] = xValues[i];
            // note: no need to copy x-values unless you implement a ToString.
            // more efficient is to simply use the xValues[] directly.

            for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
                for (int i = 0; i < numInput; ++i)
                    hSums[j] += this.inputs[i] * this.ihWeights[i][j]; // note +=

            for (int i = 0; i < numHidden; ++i)  // add biases to hidden sums
                hSums[i] += this.hBiases[i];

            for (int i = 0; i < numHidden; ++i)   // apply activation
                this.hOutputs[i] = HyperTan(hSums[i]); // hard-coded

            for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
                for (int i = 0; i < numHidden; ++i)
                    oSums[j] += hOutputs[i] * hoWeights[i][j];

            for (int i = 0; i < numOutput; ++i)  // add biases to output sums
                oSums[i] += oBiases[i];

            double[] softOut = Softmax(oSums); // all outputs at once for efficiency
            Array.Copy(softOut, outputs, softOut.Length);

            double[] retResult = new double[numOutput]; // could define a GetOutputs 
            Array.Copy(this.outputs, retResult, retResult.Length);
            return retResult;
        }

        private static double HyperTan(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }

        private static double[] Softmax(double[] oSums)
        {
            // does all output nodes at once so scale
            // doesn't have to be re-computed each time

            double sum = 0.0; // Determine max output sum. 
            for (int i = 0; i < oSums.Length; ++i)
                sum += Math.Exp(oSums[i]);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i]) / sum;

            return result; // now scaled so that xi sum to 1.0
        }

        /*
        private static double[] Softmax(double[] oSums) //private static double[] Softmax(double[] oSums)
        { // Does all output nodes at once so scale doesn't have to be re-computed each time. 
            double max = oSums[0]; // Determine max output sum. 
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max) max = oSums[i]; // Determine scaling factor -- sum of exp(each val - max). 
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);
            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale;
            return result; // Now scaled so that xi sum to 1.0. 
        }

            //ver http://stackoverflow.com/questions/34968722/softmax-function-python
        */

        public double[] Train(double[][] trainData, int maxEpochs,
          double learnRate, double momentum, string mode)
        {
            // train using back-prop
            // back-prop specific arrays
            double[][] hoGrads = MakeMatrix(numHidden, numOutput, 0.0); // hidden-to-output weight gradients
            double[] obGrads = new double[numOutput];                   // output bias gradients

            double[][] ihGrads = MakeMatrix(numInput, numHidden, 0.0);  // input-to-hidden weight gradients
            double[] hbGrads = new double[numHidden];                   // hidden bias gradients

            double[] oSignals = new double[numOutput];                  // local gradient output signals - gradients w/o associated input terms
            double[] hSignals = new double[numHidden];                  // local gradient hidden node signals

            // back-prop momentum specific arrays 
            double[][] ihPrevWeightsDelta = MakeMatrix(numInput, numHidden, 0.0);
            double[] hPrevBiasesDelta = new double[numHidden];
            double[][] hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput, 0.0);
            double[] oPrevBiasesDelta = new double[numOutput];

            int epoch = 0;
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // target values
            double derivative = 0.0;
            double errorSignal = 0.0;

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            int errInterval = maxEpochs / 10; // interval to check error

            if (mode == "mean_squared_error") {
            while (epoch < maxEpochs)
            {
                ++epoch;

                if (epoch % errInterval == 0 && epoch < maxEpochs)
                {
                    double trainErr = MeanSquaredError(trainData);
                    //Console.WriteLine("epoch = " + epoch + "  error = " + trainErr.ToString("F4"));
                    //Console.ReadLine();
                }

                Shuffle(sequence); // visit each training data in random order
                for (int ii = 0; ii < trainData.Length; ++ii)
                {
                    int idx = sequence[ii];
                    Array.Copy(trainData[idx], xValues, numInput);
                    Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
                    ComputeOutputs(xValues); // copy xValues in, compute outputs 


                   //this could be a separate method
                  // UpdateWeights(tValues, learnRate, momentum, mode); // Find better weights. 
                  // indices: i = inputs, j = hiddens, k = outputs

                        // 1. compute output node signals (assumes softmax)
                        for (int k = 0; k < numOutput; ++k)
                    {
                        errorSignal = tValues[k] - outputs[k];  // Wikipedia uses (o-t)
                        derivative = (1 - outputs[k]) * outputs[k]; // for softmax
                        oSignals[k] = errorSignal * derivative;
                    }

                    // 2. compute hidden-to-output weight gradients using output signals
                    for (int j = 0; j < numHidden; ++j)
                        for (int k = 0; k < numOutput; ++k)
                            hoGrads[j][k] = oSignals[k] * hOutputs[j];

                    // 2b. compute output bias gradients using output signals
                    for (int k = 0; k < numOutput; ++k)
                        obGrads[k] = oSignals[k] * 1.0; // dummy assoc. input value

                    // 3. compute hidden node signals
                    for (int j = 0; j < numHidden; ++j)
                    {
                        derivative = (1 + hOutputs[j]) * (1 - hOutputs[j]); // for tanh
                        double sum = 0.0; // need sums of output signals times hidden-to-output weights
                        for (int k = 0; k < numOutput; ++k)
                        {
                            sum += oSignals[k] * hoWeights[j][k]; // represents error signal
                        }
                        hSignals[j] = derivative * sum;
                    }

                    // 4. compute input-hidden weight gradients
                    for (int i = 0; i < numInput; ++i)
                        for (int j = 0; j < numHidden; ++j)
                            ihGrads[i][j] = hSignals[j] * inputs[i];

                    // 4b. compute hidden node bias gradients
                    for (int j = 0; j < numHidden; ++j)
                        hbGrads[j] = hSignals[j] * 1.0; // dummy 1.0 input

                    // == update weights and biases

                    // update input-to-hidden weights
                    for (int i = 0; i < numInput; ++i)
                    {
                        for (int j = 0; j < numHidden; ++j)
                        {
                            double delta = ihGrads[i][j] * learnRate;
                            ihWeights[i][j] += delta; // would be -= if (o-t)
                            ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
                            ihPrevWeightsDelta[i][j] = delta; // save for next time
                        }
                    }

                    // update hidden biases
                    for (int j = 0; j < numHidden; ++j)
                    {
                        double delta = hbGrads[j] * learnRate;
                        hBiases[j] += delta;
                        hBiases[j] += hPrevBiasesDelta[j] * momentum;
                        hPrevBiasesDelta[j] = delta;
                    }

                    // update hidden-to-output weights
                    for (int j = 0; j < numHidden; ++j)
                    {
                        for (int k = 0; k < numOutput; ++k)
                        {
                            double delta = hoGrads[j][k] * learnRate;
                            hoWeights[j][k] += delta;
                            hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
                            hoPrevWeightsDelta[j][k] = delta;
                        }
                    }

                    // update output node biases
                    for (int k = 0; k < numOutput; ++k)
                    {
                        double delta = obGrads[k] * learnRate;
                        oBiases[k] += delta;
                        oBiases[k] += oPrevBiasesDelta[k] * momentum;
                        oPrevBiasesDelta[k] = delta;
                    }

                } // each training item

            } // while
            }//mode==MSE
            else if (mode== "mean_cross_entropy_error")
            {
            }
            double[] bestWts = GetWeights();
            return bestWts;
        } // Train

       /* public void UpdateWeights(double[] tValues, double learnRate, double momentum, String mode)
        {
            // Update the weights and biases using back-propagation. 
            // Assumes that SetWeights and ComputeOutputs have been called 
            // and matrices have values (other than 0.0). 
            if (tValues.Length != numOutput) throw new Exception("target values not same Length as output in UpdateWeights");

            // 1. Compute output gradients. 
            if (mode == "mean_squared_error")
            {
                for (int i = 0; i < numOutput; ++i)
                {
                    // Derivative for softmax = (1 - y) * y (same as log-sigmoid). 
                    double derivative = (1 - outputs[i]) * outputs[i];
                    oGrads[i] = derivative * (tValues[i] - outputs[i]);// the 'Mean squared error' version includes this (1-y)(y) derivative. 
                }
            }
            else if (mode == "mean_cross_entropy_error")
            {
                for (int i = 0; i < numOutput; ++i)
                {
                    oGrads[i] = tValues[i] - outputs[i]; // Assumes softmax. 
                }
            }
            // 2. Compute hidden gradients. 
            for (int i = 0; i < numHidden; ++i)
            {
                // Derivative of tanh = (1 - y) * (1 + y). 
                double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]);
                double sum = 0.0;
                for (int j = 0; j < numOutput; ++j) // Each hidden delta is the sum of numOutput terms. 
                { double x = oGrads[j] * hoWeights[i][j]; sum += x; }
                hGrads[i] = derivative * sum;
            } // 3a. Update hidden weights (gradients must be computed right-to-left but weights 
              // can be updated in any order). 
            for (int i = 0; i < numInput; ++i) // 0..2 (3) 
            {
                for (int j = 0; j < numHidden; ++j)
                // 0..3 (4) 
                {
                    double delta = learnRate * hGrads[j] * inputs[i];
                    // Compute the new delta. 
                    ihWeights[i][j] += delta;
                    // Update -- note '+' instead of '-'. 
                    // Now add momentum using previous delta.
                    ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j];
                    ihPrevWeightsDelta[i][j] = delta;
                    // Don't forget to save the delta for momentum . 
                }
            }
            // 3b. Update hidden biases. 
            for (int i = 0; i < numHidden; ++i)
            {
                double delta = learnRate * hGrads[i];
                // 1.0 is constant input for bias. 
                hBiases[i] += delta;
                hBiases[i] += momentum * hPrevBiasesDelta[i];
                // Momentum. 
                hPrevBiasesDelta[i] = delta;
                // Don't forget to save the delta. 
            }
            // 4. Update hidden-output weights. 
            for (int i = 0; i < numHidden; ++i)
            {
                for (int j = 0; j < numOutput; ++j)
                {
                    double delta = learnRate * oGrads[j] * hOutputs[i];
                    hoWeights[i][j] += delta;
                    hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j];
                    // Momentum. 
                    hoPrevWeightsDelta[i][j] = delta;
                    // Save. 
                }
            }
            // 4b. Update output biases. 
            for (int i = 0; i < numOutput; ++i)
            {
                double delta = learnRate * oGrads[i] * 1.0;
                oBiases[i] += delta;
                oBiases[i] += momentum * oPrevBiasesDelta[i]; // Momentum. 
                oPrevBiasesDelta[i] = delta;
                // save 
            }
        }
        // UpdateWeights 
*/
        private void Shuffle(int[] sequence) // instance method
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = this.rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        } // Shuffle

        //*******************************************Ways of calculating the error****************************
        //***************************************************************************************************
        private double MeanSquaredError(double[][] trainData)  //MSE
        { // Average squared error per training item. 

            double sumSquaredError = 0.0;
            double[] xValues = new double[numInput]; // First numInput values in trainData. 
            double[] tValues = new double[numOutput]; // Last numOutput values. 
                                                      
            for (int i = 0; i < trainData.Length; ++i)  // for each training case.
            {
                Array.Copy(trainData[i], xValues, numInput);
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // Get target values. 
                double[] yValues = this.ComputeOutputs(xValues); // Outputs using current weights. 
                for (int j = 0; j < numOutput; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }
            return sumSquaredError / trainData.Length;
        }

        private double MeanCrossEntropyError(double[][] trainData) //MCEE
        {
            double sumError = 0.0;
            double[] xValues = new double[numInput]; // First numInput values in trainData. 
            double[] tValues = new double[numOutput]; // Last numOutput values. 

            for (int i = 0; i < trainData.Length; ++i) // Training data: (6.9 3.2 5.7 2.3) (0 0 1). 
            {
                Array.Copy(trainData[i], xValues, numInput); // Get xValues. 
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // Get target values. 
                double[] yValues = this.ComputeOutputs(xValues); // Compute output using current weights.
                for (int j = 0; j < numOutput; ++j)
                {
                    sumError += Math.Log(yValues[j]) * tValues[j]; // CE error for one training data. 
                }
            }
            return -1.0 * sumError / trainData.Length;
        }


        //*****************************************MEASURES OF PERFORMANCE*************************************
        //*****************************************************************************************************

        //relembrar: meço diretamente o true positive(TP), false negative (FN), false positive (FP) e true negative (TN)
        //o numero de eventos positivos (Pos)=TP+FN, o de negativos =TN+FP
        //o numero de observações positivas =TP+FP e o de negativas=TN+FN

        //***********************************************accuracy using winer-takes all**************************
        //******************************************************************************************************
        public double Accuracy(double[][] testData, string mode) //ACC=(TP+TN)/Teventos
        {
            // percentage correct using winner-takes all
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // targets
            double[] yValues; // computed Y

            //***************************************Mostrar valores das predições****************************************************+
            //if (mode == "test")
            // Console.Write("values of predictions for the test set\n");
            //*********************************************************************************************************************

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, numInput); // get x-values
                Array.Copy(testData[i], numInput, tValues, 0, numOutput); // get t-values
                yValues = this.ComputeOutputs(xValues);

                //*******************************valores de previsoes**************************************************************
                //if (mode == "test")
                //   Console.WriteLine("sample " + i + ": " + yValues[0] + " " + yValues[1]);
                //**********************************************************************************************************

                int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?
                int tMaxIndex = MaxIndex(tValues);

                if (maxIndex == tMaxIndex)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong); // No check for divide by zero. 
        }


        public void Precision(double[][] testData) //PREC =TP/(TP+FP)
        {
           
        }

        public void Recall(double[][] testData) //TPR =TP/Eventos positivos=TP/(TP+FN)
        {

        }

        public void F1Score(double[][] testData) //F1score==2*(TPR*PREC)/(TPR+PREC)
        {

        }



        private static int MaxIndex(double[] vector) // helper for Accuracy()
        {
            // index of largest value
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i];
                    bigIndex = i;
                }
            }
            return bigIndex;
        }
        public int[] WrongCorrect(double[][] testData)
        {
            // num wrong, num correct using winner-takes all
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // targets
            double[] yValues; // computed Y

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, numInput); // parse test data into x-values and t-values
                Array.Copy(testData[i], numInput, tValues, 0, numOutput);
                yValues = this.ComputeOutputs(xValues);
                int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?

                if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y)
                    ++numCorrect;
                else
                    ++numWrong;
            }

            //Console.WriteLine("num corect, wrong = " + numCorrect + " " + numWrong);
            return new int[] { numWrong, numCorrect };
        }
    } // NeuralNetwork
} // ns

//*/
//******************************************************Iris dataset-hardcoded***************************************************
//*****************************************************************************************************************************
/* 

        double[][] allData = new double[150][];
        allData[0] = new double[] { 5.1, 3.5, 1.4, 0.2, 0, 0, 1 };
        allData[1] = new double[] { 4.9, 3.0, 1.4, 0.2, 0, 0, 1 }; // Iris setosa = 0 0 1 
        allData[2] = new double[] { 4.7, 3.2, 1.3, 0.2, 0, 0, 1 }; // Iris versicolor = 0 1 0 
        allData[3] = new double[] { 4.6, 3.1, 1.5, 0.2, 0, 0, 1 }; // Iris virginica = 1 0 0 
        allData[4] = new double[] { 5.0, 3.6, 1.4, 0.2, 0, 0, 1 };
        allData[5] = new double[] { 5.4, 3.9, 1.7, 0.4, 0, 0, 1 };
        allData[6] = new double[] { 4.6, 3.4, 1.4, 0.3, 0, 0, 1 };
        allData[7] = new double[] { 5.0, 3.4, 1.5, 0.2, 0, 0, 1 };
        allData[8] = new double[] { 4.4, 2.9, 1.4, 0.2, 0, 0, 1 };
        allData[9] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
        allData[10] = new double[] { 5.4, 3.7, 1.5, 0.2, 0, 0, 1 };
        allData[11] = new double[] { 4.8, 3.4, 1.6, 0.2, 0, 0, 1 };
        allData[12] = new double[] { 4.8, 3.0, 1.4, 0.1, 0, 0, 1 };
        allData[13] = new double[] { 4.3, 3.0, 1.1, 0.1, 0, 0, 1 };
        allData[14] = new double[] { 5.8, 4.0, 1.2, 0.2, 0, 0, 1 };
        allData[15] = new double[] { 5.7, 4.4, 1.5, 0.4, 0, 0, 1 };
        allData[16] = new double[] { 5.4, 3.9, 1.3, 0.4, 0, 0, 1 };
        allData[17] = new double[] { 5.1, 3.5, 1.4, 0.3, 0, 0, 1 };
        allData[18] = new double[] { 5.7, 3.8, 1.7, 0.3, 0, 0, 1 };
        allData[19] = new double[] { 5.1, 3.8, 1.5, 0.3, 0, 0, 1 };
        allData[20] = new double[] { 5.4, 3.4, 1.7, 0.2, 0, 0, 1 };
        allData[21] = new double[] { 5.1, 3.7, 1.5, 0.4, 0, 0, 1 };
        allData[22] = new double[] { 4.6, 3.6, 1.0, 0.2, 0, 0, 1 };
        allData[23] = new double[] { 5.1, 3.3, 1.7, 0.5, 0, 0, 1 };
        allData[24] = new double[] { 4.8, 3.4, 1.9, 0.2, 0, 0, 1 };
        allData[25] = new double[] { 5.0, 3.0, 1.6, 0.2, 0, 0, 1 };
        allData[26] = new double[] { 5.0, 3.4, 1.6, 0.4, 0, 0, 1 };
        allData[27] = new double[] { 5.2, 3.5, 1.5, 0.2, 0, 0, 1 };
        allData[28] = new double[] { 5.2, 3.4, 1.4, 0.2, 0, 0, 1 };
        allData[29] = new double[] { 4.7, 3.2, 1.6, 0.2, 0, 0, 1 };
        allData[30] = new double[] { 4.8, 3.1, 1.6, 0.2, 0, 0, 1 };
        allData[31] = new double[] { 5.4, 3.4, 1.5, 0.4, 0, 0, 1 };
        allData[32] = new double[] { 5.2, 4.1, 1.5, 0.1, 0, 0, 1 };
        allData[33] = new double[] { 5.5, 4.2, 1.4, 0.2, 0, 0, 1 };
        allData[34] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
        allData[35] = new double[] { 5.0, 3.2, 1.2, 0.2, 0, 0, 1 };
        allData[36] = new double[] { 5.5, 3.5, 1.3, 0.2, 0, 0, 1 };
        allData[37] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
        allData[38] = new double[] { 4.4, 3.0, 1.3, 0.2, 0, 0, 1 };
        allData[39] = new double[] { 5.1, 3.4, 1.5, 0.2, 0, 0, 1 };
        allData[40] = new double[] { 5.0, 3.5, 1.3, 0.3, 0, 0, 1 };
        allData[41] = new double[] { 4.5, 2.3, 1.3, 0.3, 0, 0, 1 };
        allData[42] = new double[] { 4.4, 3.2, 1.3, 0.2, 0, 0, 1 };
        allData[43] = new double[] { 5.0, 3.5, 1.6, 0.6, 0, 0, 1 };
        allData[44] = new double[] { 5.1, 3.8, 1.9, 0.4, 0, 0, 1 };
        allData[45] = new double[] { 4.8, 3.0, 1.4, 0.3, 0, 0, 1 };
        allData[46] = new double[] { 5.1, 3.8, 1.6, 0.2, 0, 0, 1 };
        allData[47] = new double[] { 4.6, 3.2, 1.4, 0.2, 0, 0, 1 };
        allData[48] = new double[] { 5.3, 3.7, 1.5, 0.2, 0, 0, 1 };
        allData[49] = new double[] { 5.0, 3.3, 1.4, 0.2, 0, 0, 1 };
        allData[50] = new double[] { 7.0, 3.2, 4.7, 1.4, 0, 1, 0 };
        allData[51] = new double[] { 6.4, 3.2, 4.5, 1.5, 0, 1, 0 };
        allData[52] = new double[] { 6.9, 3.1, 4.9, 1.5, 0, 1, 0 };
        allData[53] = new double[] { 5.5, 2.3, 4.0, 1.3, 0, 1, 0 };
        allData[54] = new double[] { 6.5, 2.8, 4.6, 1.5, 0, 1, 0 };
        allData[55] = new double[] { 5.7, 2.8, 4.5, 1.3, 0, 1, 0 };
        allData[56] = new double[] { 6.3, 3.3, 4.7, 1.6, 0, 1, 0 };
        allData[57] = new double[] { 4.9, 2.4, 3.3, 1.0, 0, 1, 0 };
        allData[58] = new double[] { 6.6, 2.9, 4.6, 1.3, 0, 1, 0 };
        allData[59] = new double[] { 5.2, 2.7, 3.9, 1.4, 0, 1, 0 };
        allData[60] = new double[] { 5.0, 2.0, 3.5, 1.0, 0, 1, 0 };
        allData[61] = new double[] { 5.9, 3.0, 4.2, 1.5, 0, 1, 0 };
        allData[62] = new double[] { 6.0, 2.2, 4.0, 1.0, 0, 1, 0 };
        allData[63] = new double[] { 6.1, 2.9, 4.7, 1.4, 0, 1, 0 };
        allData[64] = new double[] { 5.6, 2.9, 3.6, 1.3, 0, 1, 0 };
        allData[65] = new double[] { 6.7, 3.1, 4.4, 1.4, 0, 1, 0 };
        allData[66] = new double[] { 5.6, 3.0, 4.5, 1.5, 0, 1, 0 };
        allData[67] = new double[] { 5.8, 2.7, 4.1, 1.0, 0, 1, 0 };
        allData[68] = new double[] { 6.2, 2.2, 4.5, 1.5, 0, 1, 0 };
        allData[69] = new double[] { 5.6, 2.5, 3.9, 1.1, 0, 1, 0 };
        allData[70] = new double[] { 5.9, 3.2, 4.8, 1.8, 0, 1, 0 };
        allData[71] = new double[] { 6.1, 2.8, 4.0, 1.3, 0, 1, 0 };
        allData[72] = new double[] { 6.3, 2.5, 4.9, 1.5, 0, 1, 0 };
        allData[73] = new double[] { 6.1, 2.8, 4.7, 1.2, 0, 1, 0 };
        allData[74] = new double[] { 6.4, 2.9, 4.3, 1.3, 0, 1, 0 };
        allData[75] = new double[] { 6.6, 3.0, 4.4, 1.4, 0, 1, 0 };
        allData[76] = new double[] { 6.8, 2.8, 4.8, 1.4, 0, 1, 0 };
        allData[77] = new double[] { 6.7, 3.0, 5.0, 1.7, 0, 1, 0 };
        allData[78] = new double[] { 6.0, 2.9, 4.5, 1.5, 0, 1, 0 };
        allData[79] = new double[] { 5.7, 2.6, 3.5, 1.0, 0, 1, 0 };
        allData[80] = new double[] { 5.5, 2.4, 3.8, 1.1, 0, 1, 0 };
        allData[81] = new double[] { 5.5, 2.4, 3.7, 1.0, 0, 1, 0 };
        allData[82] = new double[] { 5.8, 2.7, 3.9, 1.2, 0, 1, 0 };
        allData[83] = new double[] { 6.0, 2.7, 5.1, 1.6, 0, 1, 0 };
        allData[84] = new double[] { 5.4, 3.0, 4.5, 1.5, 0, 1, 0 };
        allData[85] = new double[] { 6.0, 3.4, 4.5, 1.6, 0, 1, 0 };
        allData[86] = new double[] { 6.7, 3.1, 4.7, 1.5, 0, 1, 0 };
        allData[87] = new double[] { 6.3, 2.3, 4.4, 1.3, 0, 1, 0 };
        allData[88] = new double[] { 5.6, 3.0, 4.1, 1.3, 0, 1, 0 };
        allData[89] = new double[] { 5.5, 2.5, 4.0, 1.3, 0, 1, 0 };
        allData[90] = new double[] { 5.5, 2.6, 4.4, 1.2, 0, 1, 0 };
        allData[91] = new double[] { 6.1, 3.0, 4.6, 1.4, 0, 1, 0 };
        allData[92] = new double[] { 5.8, 2.6, 4.0, 1.2, 0, 1, 0 };
        allData[93] = new double[] { 5.0, 2.3, 3.3, 1.0, 0, 1, 0 };
        allData[94] = new double[] { 5.6, 2.7, 4.2, 1.3, 0, 1, 0 };
        allData[95] = new double[] { 5.7, 3.0, 4.2, 1.2, 0, 1, 0 };
        allData[96] = new double[] { 5.7, 2.9, 4.2, 1.3, 0, 1, 0 };
        allData[97] = new double[] { 6.2, 2.9, 4.3, 1.3, 0, 1, 0 };
        allData[98] = new double[] { 5.1, 2.5, 3.0, 1.1, 0, 1, 0 };
        allData[99] = new double[] { 5.7, 2.8, 4.1, 1.3, 0, 1, 0 };
        allData[100] = new double[] { 6.3, 3.3, 6.0, 2.5, 1, 0, 0 };
        allData[101] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
        allData[102] = new double[] { 7.1, 3.0, 5.9, 2.1, 1, 0, 0 };
        allData[103] = new double[] { 6.3, 2.9, 5.6, 1.8, 1, 0, 0 };
        allData[104] = new double[] { 6.5, 3.0, 5.8, 2.2, 1, 0, 0 };
        allData[105] = new double[] { 7.6, 3.0, 6.6, 2.1, 1, 0, 0 };
        allData[106] = new double[] { 4.9, 2.5, 4.5, 1.7, 1, 0, 0 };
        allData[107] = new double[] { 7.3, 2.9, 6.3, 1.8, 1, 0, 0 };
        allData[108] = new double[] { 6.7, 2.5, 5.8, 1.8, 1, 0, 0 };
        allData[109] = new double[] { 7.2, 3.6, 6.1, 2.5, 1, 0, 0 };
        allData[110] = new double[] { 6.5, 3.2, 5.1, 2.0, 1, 0, 0 };
        allData[111] = new double[] { 6.4, 2.7, 5.3, 1.9, 1, 0, 0 };
        allData[112] = new double[] { 6.8, 3.0, 5.5, 2.1, 1, 0, 0 };
        allData[113] = new double[] { 5.7, 2.5, 5.0, 2.0, 1, 0, 0 };
        allData[114] = new double[] { 5.8, 2.8, 5.1, 2.4, 1, 0, 0 };
        allData[115] = new double[] { 6.4, 3.2, 5.3, 2.3, 1, 0, 0 };
        allData[116] = new double[] { 6.5, 3.0, 5.5, 1.8, 1, 0, 0 };
        allData[117] = new double[] { 7.7, 3.8, 6.7, 2.2, 1, 0, 0 };
        allData[118] = new double[] { 7.7, 2.6, 6.9, 2.3, 1, 0, 0 };
        allData[119] = new double[] { 6.0, 2.2, 5.0, 1.5, 1, 0, 0 };
        allData[120] = new double[] { 6.9, 3.2, 5.7, 2.3, 1, 0, 0 };
        allData[121] = new double[] { 5.6, 2.8, 4.9, 2.0, 1, 0, 0 };
        allData[122] = new double[] { 7.7, 2.8, 6.7, 2.0, 1, 0, 0 };
        allData[123] = new double[] { 6.3, 2.7, 4.9, 1.8, 1, 0, 0 };
        allData[124] = new double[] { 6.7, 3.3, 5.7, 2.1, 1, 0, 0 };
        allData[125] = new double[] { 7.2, 3.2, 6.0, 1.8, 1, 0, 0 };
        allData[126] = new double[] { 6.2, 2.8, 4.8, 1.8, 1, 0, 0 };
        allData[127] = new double[] { 6.1, 3.0, 4.9, 1.8, 1, 0, 0 };
        allData[128] = new double[] { 6.4, 2.8, 5.6, 2.1, 1, 0, 0 };
        allData[129] = new double[] { 7.2, 3.0, 5.8, 1.6, 1, 0, 0 };
        allData[130] = new double[] { 7.4, 2.8, 6.1, 1.9, 1, 0, 0 };
        allData[131] = new double[] { 7.9, 3.8, 6.4, 2.0, 1, 0, 0 };
        allData[132] = new double[] { 6.4, 2.8, 5.6, 2.2, 1, 0, 0 };
        allData[133] = new double[] { 6.3, 2.8, 5.1, 1.5, 1, 0, 0 };
        allData[134] = new double[] { 6.1, 2.6, 5.6, 1.4, 1, 0, 0 };
        allData[135] = new double[] { 7.7, 3.0, 6.1, 2.3, 1, 0, 0 };
        allData[136] = new double[] { 6.3, 3.4, 5.6, 2.4, 1, 0, 0 };
        allData[137] = new double[] { 6.4, 3.1, 5.5, 1.8, 1, 0, 0 };
        allData[138] = new double[] { 6.0, 3.0, 4.8, 1.8, 1, 0, 0 };
        allData[139] = new double[] { 6.9, 3.1, 5.4, 2.1, 1, 0, 0 };
        allData[140] = new double[] { 6.7, 3.1, 5.6, 2.4, 1, 0, 0 };
        allData[141] = new double[] { 6.9, 3.1, 5.1, 2.3, 1, 0, 0 };
        allData[142] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
        allData[143] = new double[] { 6.8, 3.2, 5.9, 2.3, 1, 0, 0 };
        allData[144] = new double[] { 6.7, 3.3, 5.7, 2.5, 1, 0, 0 };
        allData[145] = new double[] { 6.7, 3.0, 5.2, 2.3, 1, 0, 0 };
        allData[146] = new double[] { 6.3, 2.5, 5.0, 1.9, 1, 0, 0 };
        allData[147] = new double[] { 6.5, 3.0, 5.2, 2.0, 1, 0, 0 };
        allData[148] = new double[] { 6.2, 3.4, 5.4, 2.3, 1, 0, 0 };
        allData[149] = new double[] { 5.9, 3.0, 5.1, 1.8, 1, 0, 0 };
*/


/*
why use k-fold cross validation?
A simplistic approach would be to use all of the available data items to train the neural network. 
However, this approach would likely find weights and bias values that match the data extremely well -- in fact, probably with 100 percent accuracy -- 
but when presented with a new, previously unseen set of input data, the neural network would likely predict very poorly. 
This phenomenon is called over-fitting. To avoid over-fitting, the idea is to separate the available data into a training data set 
(typically 80 percent to 90 percent of the data) that's used to find a set of good weights and bias values, and a test set 
(the remaining 10 percent to 20 percent of the data) that is used to evaluate the quality of resulting neural network.

The simplest form of cross-validation randomly separates the available data into a single training set and a single test set. 
This is called hold-out validation. But the hold-out approach is somewhat risky because an unlucky split of the available data could lead to
an ineffective neural network. One possibility is to repeat hold-out validation several times. This is called repeated sub-sampling validation. 
But this approach also entails some risk because, although unlikely, some data items could be used only for training and never for testing, or vice versa.

The idea behind k-fold cross-validation is to divide all the available data items into roughly equal-sized sets. 
Each set is used exactly once as the test set while the remaining data is used as the training set.
You choose the number of fold and each subset of data has (number of data items)/(number of folds)
The k-fold cross-validation process iterates over the number of folds.
In situations where the number of folds doesn't divide evenly, the last data subset picks up the extra data items.

In high-level pseudo-code, method CrossValidate is:
for each fold k
  instantiate a neural network
  get reference to training data for k
  get reference to test data for k
  train the neural network
  accumulate number wrong, correct
end for
return number (total wrong) / (wrong + correct)

This pseudocode is for a simple classification error -- the number of wrong classifications on the test data divided by the total number of test items 
-- as the measure of quality. An superior alternative measure is cross-entropy error.

More folds yields a more accurate estimate of classification error at the expense of time. 
An extreme approach is to set the number of folds equal to the number of available data items. 
This will result in exactly one data item being a test item during each fold iteration. This technique is called leave-one-out (LOO) cross-validation.
 */
