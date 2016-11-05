using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Labs.LabsFolder.Lab1
{
    class Perceptron
    {
        public int[,] x = new int[8, 3]; // масив вхідних даних
        public double[] weights = new double[3]; // масив ваг
        public const double rate = 0.05; //швидкість навчання
        public const double teta = 0.3; // поріг активації
        public int out1; // вихід нейрона
        public int[] sigma = new int[8]; // масив дельт похибок
        public double a; // сума входів
        public int true_result; // очікуваний результат
        public double[] delta = new double[3]; // масив дельт ваг

        public Perceptron() // конструктор 
        {
            for (int i = 0; i < weights.Length; i++)
            {
                Random rdn = new Random();
                weights[i] = rdn.NextDouble() * (1.0) + 0.0;
               
            }
        }
        
        // функція для визначення, чи нейрон активний
        public void activation(int i)
        {
            // обчислення логічної функції
            true_result = Convert.ToInt32(Convert.ToBoolean(x[i, 0]) && (!Convert.ToBoolean(x[i, 1]) || Convert.ToBoolean(x[i, 2])));
            a = x[i, 0] * weights[0] + x[i, 1] * weights[1] + x[i, 2] * weights[2];
            if (a > teta)
            {
                out1 = 1;
            }
            else
            {
                out1 = 0;
            }
            sigma[i] = true_result - out1;
            
          
        }

        // навчання нейрона
        public void study(int i)
        {
            if (out1 != true_result)
            {

                delta[0] = rate * sigma[i] * x[i, 0];
                delta[1] = rate * sigma[i] * x[i, 1];
                delta[2] = rate * sigma[i] * x[i, 2];

                weights[0] += delta[0];
                weights[1] += delta[1];
                weights[2] += delta[2];
            }

        }
    }

}

