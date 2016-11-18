using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    #region Функції активації та їхні похідні
    // enum для перелічення можливих функцій активації
    public enum TransferFunction
    {
        None,
        Sigmoid,
        BipolarSigmoid,
        Linear
    } 
    
    public static class TransferFunctions
    {
        // Функція активації
        public static double Evaluate(TransferFunction tFunc, double input)
        {
            switch (tFunc)
            {
                case TransferFunction.Sigmoid:
                    return sigmoid(input);
                case TransferFunction.BipolarSigmoid:
                    return bipolarsigmoid(input);
                case TransferFunction.Linear:
                    return linear(input);
                case TransferFunction.None :
                default:
                    return 0.0;
            }
        }

        // Похідна функції активації
        public static double DerivativeEvaluate(TransferFunction tFunc, double input)
        {
            switch (tFunc)
            {
                case TransferFunction.Linear:
                    return linear_derivative(input);
                case TransferFunction.Sigmoid:
                    return sigmoid_derivative(input);
                case TransferFunction.BipolarSigmoid:
                    return bipolarsigmoid_derivative(input);
                case TransferFunction.None :
                default:
                    return 0.0;

            }

        }

        // лінійна функція
        private static double linear(double x)
        {
            return x;
        }

        // похідна лінійної функції
        private static double linear_derivative(double x)
        {
            return 1.0;
        }

        // Сигмоїда
        private static double sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        // Похідна сигмоїди
        private static double sigmoid_derivative(double x)
        {
            return sigmoid(x) * (1 - sigmoid(x));
        }

        private static double bipolarsigmoid(double x)
        {
            return 2.0 / (1.0 + Math.Exp(-x)) - 1;
        }

        private static double bipolarsigmoid_derivative(double x)
        {
            return 0.5 * (1 + bipolarsigmoid(x)) * (1 - bipolarsigmoid(x));
        }

    }
    #endregion

    public class BackPropagationNetwork
    {
        #region Поля
        public int layerCount; // прихований шар + вихідний шар
        public int inputSize; // к-сть нейронів у вхдному шарі
        public int[] layerSize; // величини к-сті нейронів у прихованому та вихідному шарах 
        private TransferFunction[] transferFunction; // масив функцій активації

        public double[][] layerOtput; // вихідні дані шару
        public double[][] layerInput; // вхідні дані шару
        private double[][] bias; // відхилення
        private double[][] delta; // дельта помилки
        private double[][] previosBiasDelta; // дельта попереднього відхилення

        private double[][][] weight; // ваги, де [0] - шар, [1] - попередній нейрон, [2] - поточний нейрон
        private double[][][] previousWeightDelta; // дельта попередньої ваги


        #endregion

        #region Конструктор
        public BackPropagationNetwork(int[] layerSizes, TransferFunction[] TransferFunctions)
        {
            // Перевірка вхідних даних
            if (TransferFunctions.Length != layerSizes.Length || TransferFunctions[0] != TransferFunction.None)
                throw new ArgumentException("The network cannot be created with these parameters");
            
            // Ініціалізація шарів мережі
            layerCount = layerSizes.Length - 1;
            inputSize = layerSizes[0];
            layerSize = new int[layerCount];
            transferFunction = new TransferFunction[layerCount];

            for (int i = 0; i<layerCount; i++) 
                layerSize[i] = layerSizes[i + 1];

            for (int i = 0; i < layerCount; i++)
                transferFunction[i] = TransferFunctions[i + 1];

            // Визначення вимірів масивів
            bias = new double[layerCount][];
            previosBiasDelta = new double[layerCount][];
            delta = new double[layerCount][];
            layerOtput = new double[layerCount][];
            layerInput = new double[layerCount][];

            weight = new double[layerCount][][];
            previousWeightDelta = new double[layerCount][][];

            // Заповнення двовимірних масивів
            for (int l = 0; l<layerCount; l++)
            {
                bias[l] = new double[layerSize[l]];
                previosBiasDelta[l] = new double[layerSize[l]];
                delta[l] = new double[layerSize[l]];
                layerOtput[l] = new double[layerSize[l]];
                layerInput[l] = new double[layerSize[l]];

                weight[l] = new double[l == 0 ? inputSize : layerSize[l-1]][];
                previousWeightDelta[l] = new double[l == 0 ? inputSize : layerSize[l-1]][];

                for (int i = 0; i<(l == 0 ? inputSize : layerSize[l - 1]); i++)
                {
                    weight[l][i] = new double[layerSize[l]];
                    previousWeightDelta[l][i] = new double[layerSize[l]];
                }
            }

            // Ініціалізація ваг
            for(int l =0; l < layerCount; l++)
            {
                for(int i = 0; i < layerSize[l]; i++)
                {
                    bias[l][i] = Gaussian.GetRandomGaussian();
                    previosBiasDelta[l][i] = 0.0;
                    layerInput[l][i] = 0.0;
                    layerOtput[l][i] = 0.0;
                    delta[l][i] = 0.0;
                }

                for(int i = 0; i< (l == 0 ? inputSize : layerSize[l - 1]); i++)
                {
                    for (int j = 0; j < layerSize[l]; j++) {
                        weight[l][i][j] = Gaussian.GetRandomGaussian();
                        previousWeightDelta[l][i][j] = 0.0;
                    }
                }
            }
        }

        #endregion

        #region Methods
        public void Run(ref double[] input, out double[] output)
        {
            // Перевірка, чи введені дані відповідають кількості нейронів у вхідному шарі
            if (input.Length != inputSize)
                throw new ArgumentException("Input data isn't of the correct dimension");

            // Вихідне значення функції
            output = new double[layerSize[layerCount-1]];

            // Нормалізація вхідних значень
            double max = input.Max();
           
            // Запуск мережі
            for(int l = 0; l<layerCount; l++)
            {
                for(int j = 0; j < layerSize[l]; j++)
                {
                    double sum = 0.0;
                   
                    for(int i = 0; i<(l == 0 ? inputSize : layerSize[l-1]); i++)
                        sum += weight[l][i][j] * (l == 0 ? input[i] : layerOtput[l-1][i]);

                    sum += bias[l][j];
                    layerInput[l][j] = sum;

                    /*if (l == layerCount - 1)
                        layerOtput[l][j] = sum;
                    else*/
                    layerOtput[l][j] = TransferFunctions.Evaluate(transferFunction[l], sum);   
           
                }
            }

            // копіюємо вихід мережі у вихідний масив
            for(int i = 0; i < layerSize[layerCount-1]; i++)
            {
                output[i] = layerOtput[layerCount - 1][i];
            }

        }

        // Функція навчання
        public double Train(ref double[] input, ref double[] desired, double TrainingRate, double Momentum)
        {
            // Перевірка вхідних параметрів
            if (input.Length != inputSize)
                throw new ArgumentException("Invalid input parameter", "input");

            if (desired.Length != layerSize[layerCount - 1])
                throw new ArgumentException("Invalid input parameter", "desired");

            // Локальні змінні
            double error = 0.0, sum = 0.0, weigtdelta = 0.0, biasDelta = 0.0;
            double[] output = new double[layerSize[layerCount-1]];

            // Запуск мережі
            Run(ref input, out output);

            //Розмножуємо похибку у зворотньму порядку
            for (int l = layerCount - 1; l>=0; l--)
            {
                //Вихідний шар
                if(l == layerCount - 1)
                {
                    for (int k = 0; k < layerSize[l]; k++)
                    {
                        delta[l][k] = output[k] - desired[k];
                        error += Math.Pow(delta[l][k], 2);
                        delta[l][k] *= TransferFunctions.DerivativeEvaluate(transferFunction[l], layerInput[l][k]);
                    }
                   
                }
                //Прихований шар
                else
                {
                    for (int i =0; i<layerSize[l]; i++)
                    {
                        sum = 0.0;
                        for (int j = 0; j < layerSize[l+1]; j++)
                        {
                            sum += weight[l + 1][i][j] * delta[l+1][j];
                        }
                        sum *= TransferFunctions.DerivativeEvaluate(transferFunction[l], layerInput[l][i]);
                        delta[l][i] = sum;
                    }
                }
            }

            // Оновлення ваг та відхилень
            for (int l = 0; l<layerCount; l++)
                for (int i = 0; i < (l == 0 ? inputSize : layerSize[l-1]); i++)
                    for (int j = 0; j < layerSize[l]; j++)
                    {
                        weigtdelta = TrainingRate * delta[l][j] * (l == 0 ? input[i] : layerOtput[l - 1][i]) + Momentum * previousWeightDelta[l][i][j];
                        weight[l][i][j] -= weigtdelta;

                        previousWeightDelta[l][i][j] = weigtdelta;
                    }

            for(int l =0; l < layerCount; l++)
                for(int i = 0; i < layerSize[l]; i++)
                {
                    biasDelta = TrainingRate * delta[l][i] + Momentum * previosBiasDelta[l][i];
                    bias[l][i] -= biasDelta;

                    previosBiasDelta[l][i] = biasDelta;
                }

            return error;     
        }

        // Функція повернення ваг
        public double[][] GetWeights(int layer)
        {
            double[][] array = new double[layer == 0 ? inputSize : layerSize[layer-1]][];

            for (int i = 0; i < (layer == 0 ? inputSize : layerSize[layer - 1]); i++)
            {
                array[i] = new double[layerSize[layer]];
            }

            for(int i = 0; i < (layer == 0 ? inputSize : layerSize[layer-1]); i++)
            {
                for (int j = 0; j < layerSize[layer]; j++)
                    array[i][j] = weight[layer][i][j];
            }

            return array;
        }
        #endregion
    }

    public class SOM
    { 
        private Neuron[][] outputs; // Квадратна решітка нейронів у шарі Кохонена
        private int iteration; // Поточна ітерація
        private int length; // Величина бічної сторони решітки
        private int dimensions; // Кількість вимірів у вхідних даних
        private Random rnd = new Random(); // об'єкт Random для випадкового вибору навч. елементу
        private int[] clusters; // масив кластерів

        private List<string> names = new List<string>(); // список імен об'єктів навч. вибірки
        private List<double[]> inputs = new List<double[]>(); // список вхідних параметрів об'єктів
      
        // Конструктор
        public SOM(int length, double[][] inputs)
        {
            this.length = length;
            this.clusters = new int[length];
            for (int i = 0; i < length; i++)
                clusters[i] = i + 1;
            this.dimensions = inputs[0].Length;
            Initialise();
            LoadData(inputs);
            NormalizeInputs();
            Train(0.001);
        }

        // Ініціалізація 
        private void Initialise()
        {
            outputs = new Neuron[length][]; // вихідні нейрони
            for(int i = 0; i < length; i++)
            {
                outputs[i] = new Neuron[length];
            }

            for(int i = 0; i < length; i++)
            {
                for(int j = 0; j < length; j++)
                {
                    outputs[i][j] = new Neuron(i, j, length);
                    outputs[i][j].Weights = new double[dimensions]; 

                    for(int k = 0; k < dimensions; k++)
                    {
                        outputs[i][j].Weights[k] = rnd.NextDouble();
                    }
                }

            }
        }

        private void LoadData(double[][] inputs)
        {
            for(int i =0; i < inputs.GetUpperBound(0)+1; i++)
            {
                //this.names.Add(names[i]);
                this.inputs.Add(inputs[i]);
            }
        }

        // Нормалізація вхідних даних
        public void NormalizeInputs()
        {
            for(int j = 0; j < dimensions; j++)
            {
                double sum = 0.0;
                for(int i = 0; i < inputs.Count; i++)
                {
                    sum += inputs[i][j];
                }
                double average = sum / inputs.Count;
                for(int i =0; i < inputs.Count; i++)
                {
                    inputs[i][j] = inputs[i][j] / average;
                }
            }
        }

        // Навчання мережі
        private void Train(double maxError)
        {
            double currentError = double.MaxValue;
            while (currentError > maxError) 
            {
                currentError = 0;
                List<double[]> TrainingSet = new List<double[]>();
        
                foreach (double[] input in inputs)
                {
                    TrainingSet.Add(input);
                }

                for (int i = 0; i < inputs.Count; i++)
                {
                    double[] input = TrainingSet[rnd.Next(inputs.Count - i)];
                    currentError += TrainInput(input);
                    TrainingSet.Remove(input);
                }
            }
        }

        // Навчання усіх нейронів мережі шаблону
        private double TrainInput(double[] input)
        {
            double error = 0.0;
            Neuron bmu = FindBMU(input);
            for(int i = 0; i < length; i++)
            {
                for(int j = 0; j < length; j++)
                {
                    error += outputs[i][j].UpdateWeights(input, bmu, iteration);
                }
            }
            iteration++;

            return Math.Abs(error / (length * length));
        }

        // Визначення нейрона-переможця
        private Neuron FindBMU(double[] input)
        {
            Neuron bmu = null;
            double min = double.MaxValue;
            for(int i = 0; i < length; i++)
            {
                for(int j =0; j < length; j++)
                {
                    double distance = Distance(input, outputs[i][j].Weights);

                    if(distance < min)
                    {
                        min = distance;
                        bmu = outputs[i][j];
                    }
                }
            }
            return bmu;
        }

        // Визначення відстані від шаблону до нейрона
        private double Distance(double[] example1, double[] example2)
        {
            double value = 0;
            for(int i = 0; i < example1.Length; i++)
            {
                value += Math.Pow(example1[i] - example2[i], 2);
            }
            return Math.Sqrt(value);
        }

        // Тест мережі для навчальної множини
        public int[][] Result()
        {
            int[][] result = new int[inputs.Count][];
            for(int i =0; i < inputs.Count; i++)
            {
                result[i] = new int[3];
            }
            for(int i = 0; i < inputs.Count; i++)
            {
                Neuron n = FindBMU(inputs[i]);
                result[i][0] = n.X;
                result[i][1] = n.Y;
                result[i][2] = n.Y + 1;
            }

           return result;
        }

        // Тестування мережі для одного об'єкт
        public int[] Result(double[] example)
        {
            int[] result = new int[3];

            Neuron n = FindBMU(example);
            result[0] = n.X;
            result[1] = n.Y;
            result[2] = n.Y + 1;

            return result; 
        }


    }

    #region Допоміжний клас Нейрон для карти Кохонена
    public class Neuron
    {
        public double[] Weights; // масив ваг нейрона у шарі Кохонена
        public int X; // координата x  нейрона у решітці СОМ
        public int Y; // координата y нейрона у решітці СОМ
        private int length; // довжина квадратної решітки СОМ
        private double nf; // константа, яка залежить від сторони решітки
        
        // Конструктор для створення нейрона
        public Neuron(int x, int y, int length)
        {
            X = x;
            Y = y;
            this.length = length;
            nf = 1000 / Math.Log(length);
        }  

        // Функція Гауса для визначення міри сусідства
        private double Gauss(Neuron bmu, int iteration) // bmu - best matching unit
        {
            double distance = Math.Sqrt(Math.Pow(this.X - bmu.X, 2) 
                + Math.Pow(this.Y - bmu.Y, 2));
            return Math.Exp(-Math.Pow(distance, 2) / (Math.Pow(Strength(iteration), 2)));
        }

        // Функція Strength яка зменшує міру сусідства із часом
        public double Strength(int it)
        {
            return Math.Exp(-it / nf) * length;
        }

        // Функція зменшення швидкості навчання
        public double Rate(int iteration)
        {
            return Math.Exp(-iteration / 1000) * 0.1;
        }

        // Функція оновлення ваг
        public double UpdateWeights(double[] input, Neuron bmu, int iteration)
        {
            double sum = 0;
            for(int i = 0; i < Weights.Length; i++)
            {
                double weightdelta = Rate(iteration) * Gauss(bmu, iteration) * (input[i] - Weights[i]);
                Weights[i] += weightdelta;
                sum += weightdelta;
            }
            return sum / Weights.Length;
        }
    }
    #endregion

    // Клас для створення випадкових чисел з нормальним розподілом
    public static class Gaussian
    {
        private static Random gen = new Random();

        public static double GetRandomGaussian()
        {
            return GetRandomGaussian(0.0, 1.0);
        }

        public static double GetRandomGaussian(double mean, double stddev)
        {
            double rVal1, rVal2;
            GetRandomGaussian(mean, stddev, out rVal1, out rVal2);
            return rVal1;
        }

        public static void GetRandomGaussian(double mean, double stddev, out double val1, out double val2)
        {
            double u, v, s, t;

            do
            {
                u = 2 * gen.NextDouble() - 1;
                v = 2 * gen.NextDouble() - 1;
            } while (u * u + v * v > 1 || (u == 0 && v == 0));

            s = u * u + v * v;
            t = Math.Sqrt((-2.0 * Math.Log(s)) / s);

            val1 = stddev * t * u + mean;
            val2 = stddev * t * v + mean;


        }
    }
}
