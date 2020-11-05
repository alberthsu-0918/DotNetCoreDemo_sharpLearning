using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpLearning.RandomForest.Learners;
using SharpLearning.Containers.Matrices;
using Accord.Statistics;
using MathNet.Filtering;

namespace SharpLearningDemo
{
    public enum FeaturesSelc
    {
        mean = 0,
        std = 1,
        var = 2,
        min = 3,
        max = 4,
        median = 5,
        skew = 6,
        kuri = 7,
        crestFactor = 8,
        impulseFactor = 9,
        entropy = 10
    }

    public class FeatureFilterSelection
    {
        public FeatureFilterSelection(FeaturesSelc aFeaturesSelc, OnlineFilter aOnlineFilter)
        {
            feature = aFeaturesSelc;
            bandpass = aOnlineFilter;
        }

        public FeaturesSelc feature { get; set; }
        public OnlineFilter bandpass { get; set; }
    }


    public class Data
    {
        public double[] data;

        public void Scaling(double a, double b)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = a* data[i] + b;
            }
        }

        public void Normalize()
        {
            double vp = (data.Max() - data.Min()) / 2;
            Scaling(1.0 / vp, 0);
        }

        public double[] FeatureExtract(List<FeatureFilterSelection> selectedFeat)
        {
            double[] features = new double[selectedFeat.Count];
            int i = 0;
            foreach (FeatureFilterSelection sel in selectedFeat)
            {
                double[] filteredArray = sel.bandpass.ProcessSamples(data);

                switch (sel.feature)
                {
                    case FeaturesSelc.mean:
                        features[i] = Measures.Mean(filteredArray);
                        break;

                    case FeaturesSelc.std:
                        features[i] = Measures.StandardDeviation(filteredArray);
                        break;

                    case FeaturesSelc.var:
                        features[i] = Measures.Variance(filteredArray);
                        break;

                    case FeaturesSelc.min:
                        features[i] = filteredArray.Min();
                        break;

                    case FeaturesSelc.max:
                        features[i] = filteredArray.Max();
                        break;

                    case FeaturesSelc.median:
                        features[i] = filteredArray.Median();
                        break;

                    case FeaturesSelc.skew:
                        features[i] = Measures.Skewness(filteredArray);
                        break;

                    case FeaturesSelc.kuri:
                        features[i] = Measures.Kurtosis(filteredArray);
                        break;
                    case FeaturesSelc.crestFactor:
                        double vp = (filteredArray.Max() - filteredArray.Min()) / 2;
                        features[i] = vp / Measures.StandardDeviation(filteredArray);
                        break;
                    case FeaturesSelc.impulseFactor:
                        var absMean = filteredArray.Select(n => Math.Abs(n)).ToList();
                        features[i] = filteredArray.Max() / Measures.Mean(absMean.ToArray());
                        break;
                    case FeaturesSelc.entropy:
                        features[i] = Measures.Entropy(filteredArray);
                        break;
                }
                i++;
            }
            return features;
        }
    }

    public class LabeledData:Data
    {
        public double label;
    }

    class Program
    {
        static List<LabeledData> data_label_train = new List<LabeledData>();
        static List<LabeledData> data_label_test = new List<LabeledData>();
        static List<FeatureFilterSelection> selectedFeat = new List<FeatureFilterSelection>();

        static List<LabeledData> loadTrainingData(string filename)
        {
            List<LabeledData> loaddata = new List<LabeledData>();

            try
            {
                using (StreamReader sr = new StreamReader(filename))
                {
                    string currentLine;
                    // currentLine will be null when the StreamReader reaches the end of file
                    while ((currentLine = sr.ReadLine()) != null)
                    {
                        if (!currentLine.Contains("id"))//第一行不要
                        {
                            var temp = Array.ConvertAll(currentLine.Split(','), Double.Parse); //currentLine.Split(',');
                            loaddata.Add(new LabeledData() { label = temp.Last(), data = temp.Skip(1).Take(temp.Length - 2).ToArray() });
                        }
                    }
                }
                Console.WriteLine("Read {0} lines", loaddata.Count);

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
            //Console.ReadKey();

            return loaddata;
        }

        static List<LabeledData> loadTestData(string testFileName, string ansFileName)
        {
            List<LabeledData> loaddata = new List<LabeledData>();

            try
            {
                using (StreamReader sr = new StreamReader(testFileName))
                {
                    string currentLine;
                    // currentLine will be null when the StreamReader reaches the end of file
                    while ((currentLine = sr.ReadLine()) != null)
                    {
                        if (!currentLine.Contains("id"))//第一行不要
                        {
                            var temp = Array.ConvertAll(currentLine.Split(','), Double.Parse); //currentLine.Split(',');
                            loaddata.Add(new LabeledData() { label = temp.Last(), data = temp.Skip(1).Take(temp.Length - 2).ToArray() });
                        }
                    }
                }

                List<double> labels = new List<double>();
                using (StreamReader sr = new StreamReader(ansFileName))
                {
                    string currentLine;
                    // currentLine will be null when the StreamReader reaches the end of file
                    while ((currentLine = sr.ReadLine()) != null)
                    {
                        if (!currentLine.Contains("id"))//第一行不要
                        {
                            var temp = Array.ConvertAll(currentLine.Split(','), Double.Parse); //currentLine.Split(',');
                            labels.Add(temp.Last());
                        }
                    }
                }

                if (labels.Count == loaddata.Count)
                {
                    for (int i = 0; i < labels.Count; i++)
                        loaddata[i].label = labels[i];
                }
                else
                {
                    throw new Exception("Test data size unmatch result size");
                }
                Console.WriteLine("Read {0} lines", loaddata.Count);

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
            //Console.ReadKey();

            return loaddata;
        }

        static F64Matrix FeatureExtract(List<LabeledData> rawdata, List<FeatureFilterSelection> selectedFeat)
        {
            F64Matrix features = new F64Matrix(rawdata.Count, selectedFeat.Count);
            for (int i = 0; i < features.RowCount; i++)
            {
                var feature = rawdata[i].FeatureExtract(selectedFeat);
                for (int j = 0; j < features.ColumnCount; j++)
                    features[i, j] = feature[j];
            }
            return features;
        }

        static void Main(string[] args)
        {

            try
            {
                //Load data
                data_label_train = loadTrainingData("train.csv");
                data_label_test = loadTestData("test_data.csv", "result.csv");

                //Normalize train data
                foreach (Data reading in data_label_train)
                {
                    reading.Normalize();
                }

                //Scaling test data and Normalize it
                Random rnd = new System.Random();
                foreach (LabeledData reading in data_label_test)
                {
                    double a = rnd.NextDouble() + 1.0;
                    double b = rnd.NextDouble();
                    reading.Scaling(a, b);
                    reading.Normalize();
                }

                double fs = 8000.0;
                //Create bandpass filters
                OnlineFilter bandpass0 = OnlineFilter.CreateLowpass(ImpulseResponse.Finite, fs, 200.0);
                OnlineFilter bandpass1 = OnlineFilter.CreateBandpass(ImpulseResponse.Finite, fs, 200.0, 2000.0);
                OnlineFilter bandpass2 = OnlineFilter.CreateBandpass(ImpulseResponse.Finite, fs, 2000.0, 3000.0);
                OnlineFilter bandpass3 = OnlineFilter.CreateBandpass(ImpulseResponse.Finite, fs, 3000.0, 3400.0);
                OnlineFilter bandpass4 = OnlineFilter.CreateHighpass(ImpulseResponse.Finite, fs, 3400.0);

                //Pre process: extract features from raw data     
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.crestFactor,  bandpass0 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.skew, bandpass0 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.kuri, bandpass0 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.std, bandpass0 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.var, bandpass0 ));

                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.crestFactor, bandpass1 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.skew, bandpass1 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.kuri, bandpass1 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.std, bandpass1 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.var, bandpass1 ));

                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.crestFactor, bandpass2 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.skew, bandpass2 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.kuri, bandpass2 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.std, bandpass2 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.var, bandpass2 ));

                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.crestFactor, bandpass3 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.skew, bandpass3 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.kuri, bandpass3 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.std, bandpass3 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.var, bandpass3 ));

                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.crestFactor, bandpass4 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.skew, bandpass4 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.kuri, bandpass4 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.std, bandpass4 ));
                selectedFeat.Add(new FeatureFilterSelection( FeaturesSelc.var, bandpass4 ));

                // learn the model
                var learner = new ClassificationRandomForestLearner(trees: 50);
                var model = learner.Learn( FeatureExtract(data_label_train, selectedFeat), data_label_train.Select(x => x.label).ToArray());

                // Test: use the model for predicting new observations
                List<double> p = new List<double>();
                foreach (LabeledData reading in data_label_test)
                {
                    var prediction = model.Predict(reading.FeatureExtract(selectedFeat));
                    p.Add(prediction);
                }

                double accuracy = 0.0;
                for (int i = 0; i < data_label_test.Count; i++)
                {
                    if (data_label_test[i].label == p[i])
                    {
                        accuracy += 1.0 / data_label_test.Count * 100;
                    }
                    Console.WriteLine("Prediction result: {0} Ideal result: {1}", p[i], data_label_test[i].label);
                }

                Console.WriteLine("Accuracy {0}", accuracy.ToString());

                Console.ReadKey();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
            Console.ReadKey();
        }


        
    }        
}
