using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Labs.LabsFolder.Lab2
{
    public partial class Testing : Form
    {
        public double[] input = new double[3];
        public Testing()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            input[0] = Convert.ToDouble(textBox1.Text);
            input[1] = Convert.ToDouble(textBox2.Text);
            input[2] = Convert.ToDouble(textBox3.Text);
            this.Hide();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            this.Close();
        }
    }
}
