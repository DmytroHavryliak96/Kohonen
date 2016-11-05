using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Npgsql;
using NeuralNetwork;

namespace Labs.LabsFolder.Lab2
{
    public partial class Form2 : Form
    {
        private DataTable dt = new DataTable();
        private BackPropagationNetwork bpn;
        private int[] layerSizes;
        private double[][] input;
        private double[][] desired;
        private TransferFunction[] TFunc;
        private double TrainingRate = 0.1;
        private double Momentum = 0.15;
        private double error;
        private DataTable errors = new DataTable();
        private DataTable function = new DataTable();
        private DataTable weightsHidden = new DataTable();
        private DataTable weightsOutput = new DataTable();
        private double[][] hidden;
        private double[][] outputWeights;

        public Form2()
        {
            InitializeComponent();
        }

        private void завантажитиToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                string connstring = "Server=127.0.0.1;Port=5432;User Id=postgres;Password=1postgres;Database=Labs;";
                NpgsqlConnection conn = new NpgsqlConnection(connstring);
                conn.Open();
                string sql = "SELECT * FROM lab2 ORDER BY id_arg";
                NpgsqlDataAdapter da = new NpgsqlDataAdapter(sql, conn);
                da.Fill(dt);
                input = new double[dt.Rows.Count][];
                desired = new double[dt.Rows.Count][];

                for (int i =0; i<dt.Rows.Count; i++) 
                {
                    input[i] = new double[dt.Columns.Count-1];
                }


                for (int i = 0; i < dt.Rows.Count; i++)
                    for (int j = 1; j < dt.Columns.Count; j++)
                        input[i][j-1] = Convert.ToDouble(dt.Rows[i][j]);

                conn.Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString());
            }

        }

        private void створитиМережуToolStripMenuItem_Click(object sender, EventArgs e)
        {
            FormCreate f3 = new FormCreate();
            this.Hide();
            f3.ShowDialog();
            layerSizes = new int[f3.layerSizes.Length];
            for(int i =0; i<f3.layerSizes.Length; i++)
            {
                layerSizes[i] = f3.layerSizes[i];
            }
           
            TFunc = new TransferFunction[f3.TFunc.Length];
            for (int i = 0; i < f3.TFunc.Length; i++)
            {
                TFunc[i] = f3.TFunc[i];
            }

            f3.Close();

            for (int i = 0; i < dt.Rows.Count; i++)
            {
                desired[i] = new double[layerSizes.Last()];
            }
           

            bpn = new BackPropagationNetwork(layerSizes, TFunc);
            this.Show();
        }

        private void навчитиToolStripMenuItem_Click(object sender, EventArgs e)
        {
            dataGridView1.DataSource = errors;
            dataGridView2.DataSource = function;
            dataGridView3.DataSource = weightsHidden;
            dataGridView4.DataSource = weightsOutput;

            switch (comboBox1.Text)
            {
                case "ln|cosx1| + tgx2 + ctgx3":
                    {
                        errors.Columns.Add("iteration");
                        errors.Columns.Add("function");
                        errors.Columns.Add("error");
                        errors.Columns.Add("output[d1]");
                        errors.Columns.Add("output[d2]");

                        double sum = 0.0;
                        for(int i = 0; i< dt.Rows.Count; i++)
                        {
                            desired[i][0] = Math.Sin(input[i][0])+ Math.Sin(input[i][1])-Math.Sin(input[i][2]);/* Math.Log(Math.Abs(Math.Cos(input[i][0]))) + Math.Tan(input[i][1]) + 1/Math.Tan(input[i][2]);/*Math.Cos(input[i][0]) + Math.Tan(input[i][1]) + 1 / Math.Tan(input[i][2]);*/
                            sum += desired[i][0];
                        }

                        function.Columns.Add("i");
                        function.Columns.Add("x1");
                        function.Columns.Add("x2");
                        function.Columns.Add("x3");
                        function.Columns.Add("d1");
                        function.Columns.Add("d2");
                        function.Columns.Add("average");

                        for (int i = 0; i< desired.GetUpperBound(0)+1; i++)
                        {
                            DataRow row = function.NewRow();
                            row["i"] = i;
                            row["x1"] = input[i][0];
                            row["x2"] = input[i][1];
                            row["x3"] = input[i][2];
                            row["d1"] = desired[i][0];
                            function.Rows.Add(row);
                        }

                        double[] output = new double[2];

                        // Навчання
                        int example = 1;
                        int h = 0;
                        double[] value = new double[3] { 4, 5, 4};
                        Random rdn = new Random();
                        do
                        {
                            error = 0.0;
                            input.Reverse();

                            for (int j = 0; j < dt.Rows.Count; j++)
                                error += bpn.Train(ref input[j], ref desired[j], TrainingRate, Momentum);
                            if (h % 100 == 0)
                            {
                                bpn.Run(ref value, out output);
                                DataRow row = errors.NewRow();
                                row["iteration"] = h;
                                row["function"] = example;
                                row["error"] = error/20;
                                row["output[d1]"] = output[0];
                                //row["output[d2]"] = output[1];
                                errors.Rows.Add(row);
                            }
                            h++;
                            if (h == 200000)
                                break;

                        } while ((error/20) > 0.00127);

                        // Виведення скоригованих ваг прихованого шару
                        hidden = bpn.GetWeights(0);
                        weightsHidden.Columns.Add("Початковий/Прихований");

                        for(int i = 0; i < hidden[0].Count(); i++)
                        {
                            weightsHidden.Columns.Add(Convert.ToString(i));
                        }

                        for (int i = 0; i < hidden.GetUpperBound(0) + 1; i++)
                        {
                            DataRow row = weightsHidden.NewRow();
                            row["Початковий/Прихований"] = i;
                            for (int j = 1; j < weightsHidden.Columns.Count; j++)
                                row[j] = hidden[i][j - 1];
                            weightsHidden.Rows.Add(row);
                        }

                        // Виведення скоригованих ваг вихідного шару
                        outputWeights = bpn.GetWeights(1);
                        weightsOutput.Columns.Add("Прихований/Вихідний");

                        for(int i = 0; i < outputWeights[0].Count(); i++)
                        {
                            weightsOutput.Columns.Add(Convert.ToString(i));
                        }

                        for(int i =0; i < outputWeights.GetUpperBound(0)+1; i++)
                        {
                            DataRow row = weightsOutput.NewRow();
                            row["Прихований/Вихідний"] = i;
                            for(int j = 1; j < weightsOutput.Columns.Count; j++)
                            {
                                row[j] = outputWeights[i][j - 1];
                            }
                            weightsOutput.Rows.Add(row);
                        }


                        break;
                    }

            }
        }

        private void провестиТестToolStripMenuItem_Click(object sender, EventArgs e)
        {
            double[] outputTest = new double[1];
            double[] inputTest = new double[3];
            Testing test = new Testing();
            test.ShowDialog();
            inputTest = test.input;
            bpn.Run(ref inputTest, out outputTest);
            test.textBox4.Text = Convert.ToString(Math.Sin(inputTest[0]) + Math.Sin(inputTest[1]) - Math.Sin(inputTest[2]));//Convert.ToString(Math.Log(Math.Abs(Math.Cos(inputTest[0]))) + Math.Tan(inputTest[1]) + 1 / Math.Tan(inputTest[2]));
            test.textBox5.Text = Convert.ToString(outputTest[0]);
            test.Show(); 
        }

        private void вихідToolStripMenuItem_Click(object sender, EventArgs e)
        {
            this.Close();
        }
    }
}
