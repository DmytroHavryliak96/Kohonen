using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Labs.LabsFolder;
using Labs.LabsFolder.Lab2;
using Labs.LabsFolder.Lab3;

namespace Labs
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            this.Hide();
            switch (comboBox1.Text)
            {
                case "Лабораторна 1":
                    {
                        Form1l f1 = new Form1l();
                        f1.ShowDialog();
                        break;
                    }
                case "Лабораторна 2":
                    {
                        Form2 f2 = new Form2();
                        f2.ShowDialog();
                        break;
                    }
                case "Лабораторна 3":
                    {
                        SomForm form = new SomForm();
                        form.ShowDialog();
                        break;
                    }
            }
            this.Show();   
        }
    }
}
