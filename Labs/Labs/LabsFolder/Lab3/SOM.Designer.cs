using System.Windows.Forms;

namespace Labs.LabsFolder.Lab3
{
    partial class SomForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.GeneratedNumbers = new System.Windows.Forms.DataGridView();
            this.label1 = new System.Windows.Forms.Label();
            this.Result = new System.Windows.Forms.DataGridView();
            this.label2 = new System.Windows.Forms.Label();
            this.StartButton = new System.Windows.Forms.Button();
            this.TestButton = new System.Windows.Forms.Button();
            this.label3 = new System.Windows.Forms.Label();
            this.insert = new System.Windows.Forms.DataGridView();
            this.Coordinates = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.Cluster = new System.Windows.Forms.TextBox();
            ((System.ComponentModel.ISupportInitialize)(this.GeneratedNumbers)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Result)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.insert)).BeginInit();
            this.SuspendLayout();
            // 
            // GeneratedNumbers
            // 
            this.GeneratedNumbers.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.GeneratedNumbers.Location = new System.Drawing.Point(123, 36);
            this.GeneratedNumbers.Name = "GeneratedNumbers";
            this.GeneratedNumbers.Size = new System.Drawing.Size(271, 150);
            this.GeneratedNumbers.TabIndex = 0;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(180, 20);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(145, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Згенеровані випадкові дані";
            // 
            // Result
            // 
            this.Result.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.Result.Location = new System.Drawing.Point(450, 36);
            this.Result.Name = "Result";
            this.Result.Size = new System.Drawing.Size(240, 150);
            this.Result.TabIndex = 2;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(511, 20);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(116, 13);
            this.label2.TabIndex = 3;
            this.label2.Text = "Результат виконання";
            // 
            // StartButton
            // 
            this.StartButton.Location = new System.Drawing.Point(19, 36);
            this.StartButton.Name = "StartButton";
            this.StartButton.Size = new System.Drawing.Size(75, 23);
            this.StartButton.TabIndex = 4;
            this.StartButton.Text = "Старт";
            this.StartButton.UseVisualStyleBackColor = true;
            this.StartButton.Click += new System.EventHandler(this.StartButton_Click);
            // 
            // TestButton
            // 
            this.TestButton.Location = new System.Drawing.Point(19, 252);
            this.TestButton.Name = "TestButton";
            this.TestButton.Size = new System.Drawing.Size(75, 47);
            this.TestButton.TabIndex = 5;
            this.TestButton.Text = "Провести тест";
            this.TestButton.UseVisualStyleBackColor = true;
            this.TestButton.Click += new System.EventHandler(this.TestButton_Click);
            // 
            // label3
            // 
            this.label3.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.label3.Location = new System.Drawing.Point(19, 215);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(700, 2);
            this.label3.TabIndex = 6;
            // 
            // insert
            // 
            this.insert.AllowUserToAddRows = false;
            this.insert.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.insert.Location = new System.Drawing.Point(179, 220);
            this.insert.Name = "insert";
            this.insert.Size = new System.Drawing.Size(146, 134);
            this.insert.TabIndex = 7;
            // 
            // Coordinates
            // 
            this.Coordinates.Location = new System.Drawing.Point(458, 252);
            this.Coordinates.Name = "Coordinates";
            this.Coordinates.ReadOnly = true;
            this.Coordinates.Size = new System.Drawing.Size(232, 20);
            this.Coordinates.TabIndex = 8;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(488, 236);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(173, 13);
            this.label4.TabIndex = 9;
            this.label4.Text = "Координати нейрона-прееможця";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(524, 299);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(94, 13);
            this.label5.TabIndex = 10;
            this.label5.Text = "Кластер нейрона";
            // 
            // Cluster
            // 
            this.Cluster.Location = new System.Drawing.Point(458, 315);
            this.Cluster.Name = "Cluster";
            this.Cluster.ReadOnly = true;
            this.Cluster.Size = new System.Drawing.Size(232, 20);
            this.Cluster.TabIndex = 11;
            // 
            // SomForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(748, 375);
            this.Controls.Add(this.Cluster);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.Coordinates);
            this.Controls.Add(this.insert);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.TestButton);
            this.Controls.Add(this.StartButton);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.Result);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.GeneratedNumbers);
            this.Name = "SomForm";
            this.Text = "SOM";
            ((System.ComponentModel.ISupportInitialize)(this.GeneratedNumbers)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Result)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.insert)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.DataGridView GeneratedNumbers;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.DataGridView Result;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Button StartButton;
        private System.Windows.Forms.Button TestButton;
        private System.Windows.Forms.Label label3;
        private DataGridView insert;
        private TextBox Coordinates;
        private Label label4;
        private Label label5;
        private TextBox Cluster;
    }
}