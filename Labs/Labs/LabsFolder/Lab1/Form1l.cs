using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Labs.LabsFolder.Lab1;
using Npgsql;

namespace Labs.LabsFolder
{
    public partial class Form1l : Form
    {
        private DataSet ds = new DataSet();
        private DataTable dt = new DataTable();
 

        public Form1l()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Perceptron p = new Perceptron();

            int w = 0;
            // занесення вхідних даних із бд 
            foreach (DataRow row in dt.Rows)
            {
                for (int j = 0; j < 3; j++)
                {
                    int z;
                    z = j + 1;
                    p.x[w, j] = Convert.ToInt32(row[Convert.ToString("x" + z)]);
                }
                w++;
            }

            // формування таблиці для виведення
            DataTable dt2 = new DataTable();
            dt2.Columns.Add("w1");
            dt2.Columns.Add("w2");
            dt2.Columns.Add("w3");
            dt2.Columns.Add("teta");
            dt2.Columns.Add("rate");
            dt2.Columns.Add("x1");
            dt2.Columns.Add("x2");
            dt2.Columns.Add("x3");
            dt2.Columns.Add("a");
            dt2.Columns.Add("out");
            dt2.Columns.Add("true result");
            dt2.Columns.Add("rate*(T-Y)");
            dt2.Columns.Add("sigma*w1");
            dt2.Columns.Add("sigma*w2");
            dt2.Columns.Add("sigma*w3");
            dt2.Columns.Add("sigma*teta");

            dataGridView1.DataSource = dt2;
            bool finish;

            do
            {
                finish = false;

                for (int i = 0; i < 8; i++)
                {
                    p.activation(i);
                    p.study(i);

                    DataRow row = dt2.NewRow();
                    row["w1"] = p.weights[0];
                    row["w2"] = p.weights[1];
                    row["w3"] = p.weights[2];
                    row["teta"] = Perceptron.teta;
                    row["rate"] = Perceptron.rate;
                    row["x1"] = p.x[i, 0];
                    row["x2"] = p.x[i, 1];
                    row["x3"] = p.x[i, 2];
                    row["a"] = p.a;
                    row["out"] = p.out1;
                    row["true result"] = p.true_result;
                    row["rate*(T-Y)"] = Perceptron.rate*(p.true_result - p.out1);
                    row["sigma*w1"] = p.sigma[i]*p.weights[0];
                    row["sigma*w2"] = p.sigma[i]*p.weights[1];
                    row["sigma*w3"] = p.sigma[i]*p.weights[2]; 
                    row["sigma*teta"] = p.sigma[i]*Perceptron.teta;
                    dt2.Rows.Add(row);
                }
                DataRow row1 = dt2.NewRow();
                dt2.Rows.Add(row1);
                

                // якщо всі дельти похибок рівні 0, то завершуєм навчання
                for (int i = 0; i < p.sigma.Length; i++)
                {
                    if (p.sigma[i] != 0)
                    {
                        finish = true;
                        break;
                    }
                    else continue;
                }
            } while (finish);


        }

        private void button2_Click(object sender, EventArgs e)
        {
            try
            {
                //  під'єднання до бд
                string connstring = "Server=127.0.0.1;Port=5432;User Id=postgres;Password=1postgres;Database=Labs;";
                NpgsqlConnection conn = new NpgsqlConnection(connstring);
                conn.Open();
                string sql = "SELECT * FROM lab1";
                NpgsqlDataAdapter da = new NpgsqlDataAdapter(sql, conn);
              //  ds.Reset();
                da.Fill(dt);
               // dt = ds.Tables[0];    
                conn.Close();
            }
            catch (Exception msg)
            {
                MessageBox.Show(msg.ToString());
                throw;
            }

        }

        private void button3_Click(object sender, EventArgs e)
        {
            // вихід із форми
            this.Close();
        }
    }
}
