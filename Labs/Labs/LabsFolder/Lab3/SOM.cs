using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NeuralNetwork;

namespace Labs.LabsFolder.Lab3
{
    public partial class SomForm : Form
    {
        private SOM somnetwork;
        private double[][] input; // таблиця даних
        private int size = 10; // Кількість рядків
        private int parameters = 5; // кількість параметрів
        private int length = 3; // сторона квадратної решітки SOM
        private DataTable GeneratedTable = new DataTable();
        private DataTable ResultTable = new DataTable();
        private DataTable ParametersTable = new DataTable();

        

        public void GenerateRandomInput()
        {
            input = new double[size][];
            for(int i = 0; i < size; i++)
            {
                input[i] = new double[parameters];
            }

            Random rdn = new Random();
            for(int i = 0; i < size; i++)
            {
                for (int j = 0; j < parameters; j++)
                    input[i][j] = rdn.NextDouble();
            }

        }

        public SomForm()
        {
            InitializeComponent();
            
        }

        private void StartButton_Click(object sender, EventArgs e)
        {
            GenerateRandomInput(); // генеруємо дані
            somnetwork = new SOM(length, input); // створюємо і вчимо мережу

            // Прив'язка таблиць до DataGriedView
            GeneratedNumbers.DataSource = GeneratedTable;
            Result.DataSource = ResultTable;
            insert.DataSource = ParametersTable;

            // Створення колонок у таблицях
            GeneratedTable.Columns.Add("ID");
            for(int i = 0; i < parameters; i++)
            {
                string str = Convert.ToString(i+1);
                GeneratedTable.Columns.Add(str);
            }

            ResultTable.Columns.Add("ID");
            ResultTable.Columns.Add("Winning Neuron coordinates");
            ResultTable.Columns.Add("Cluster");

            ParametersTable.Columns.Add("Parameters");
            
            // Додавання рядків у таблиці
            for(int i = 0; i < size; i++)
            {
                DataRow row = GeneratedTable.NewRow();
                row[0] = i + 1;
                for (int j = 0; j < parameters; j++)
                    row[j+1] = Math.Round(input[i][j], 3);
                GeneratedTable.Rows.Add(row);
            }

            int[][] result = somnetwork.Result();

            for (int i = 0; i < size; i++)
            {
                DataRow row = ResultTable.NewRow();
                row[0] = i + 1;
                row[1] = Convert.ToString(result[i][0] + "; " + result[i][1]);
                row[2] = result[i][2];
                ResultTable.Rows.Add(row);
            }

            for(int i = 0; i < parameters; i++)
            {
                DataRow row = ParametersTable.NewRow();
                ParametersTable.Rows.Add(row); 
            }
        }

        private void TestButton_Click(object sender, EventArgs e)
        {
            double[] test = new double[parameters];
            int i = 0;
            foreach(DataGridViewRow row in insert.Rows)
            {
                foreach(DataGridViewCell cell in row.Cells)
                {
                    test[i] = Convert.ToDouble(cell.Value.ToString());
                    i++;
                }
            }

            int[] output = new int[3];

            output = somnetwork.Result(test);

            Coordinates.Text = Convert.ToString(output[0] + "; " + output[1]);
            Cluster.Text = Convert.ToString(output[2]);
        }
    }
}
