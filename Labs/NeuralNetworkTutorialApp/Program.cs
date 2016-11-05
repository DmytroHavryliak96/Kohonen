using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;

namespace NeuralNetworkTutorialApp
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] layerSizes = new int[3] { 3, 4, 1 };
            TransferFunction[] TFuncs = new TransferFunction[3] {TransferFunction.None,
                                                               TransferFunction.Sigmoid,
                                                               TransferFunction.Linear};

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, TFuncs);

            double[] input = new double[] { 4, 6, 8 };//, new double[] {4, 7, 5}, new double[] {7, 4, 8}, new double[] {6, 7, 5}, new double[] {7, 7, 8}};
            double[] desired = new double[] { -0.86 };//, new double[] {0.15}, new double[] {0.72 }, new double[] {0.53 }, new double[] { 0.44 } };
            /*double[] output = new double[1];
            

            double error = 0.0;

            for(int i = 0; i < 10; i++)
            {
               
                    error = bpn.Train(ref input, ref desired, 0.15, 0.1);
                    bpn.Run(ref input, out output);
                    if (i % 1 == 0)
                        Console.WriteLine("Iteration {0}: \n\t Input {1:0.000} {2:0.000} {3:0.000} Output {4:0.000} error{5:0.000}", i, input[0], input[1], input[2], output[0], error);
                    
                
            }*/

            double[][] inputs =
            {
                new double[]{ 0.4, 11.8, 0.1 },
                new double[]{ 1.9,1.9,19.5 },
                new double[]{ 1.2,23.2,0.3 },
                new double[]{ 20.9,0.0,7.9 },
                new double[]{ 13.0,19.0,11.0 },
                new double[]{ 15.5, 2.9, 68.3 }
            };

            string[] names = new string[] {"Apples", "Avocado", "Leave", "Beef Steak", "Jam", "Brazil Nuts"};

            SOM somnetwork = new SOM(3, names, inputs);
            Console.WriteLine("----");
            string row = "Water";
            double[] water = new double[] { 0.0, 0.0, 0.0 };
            int[] result = somnetwork.Result(water);
            Console.WriteLine(row + " " + result[0] + " " + result[1] );

            Console.ReadKey();
        }
    }
}
