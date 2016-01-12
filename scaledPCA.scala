package com.spark.demo1

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Row
import com.databricks.spark.csv._
import org.apache.spark.mllib.rdd
import org.apache.spark.rdd
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.ml.feature.PCA
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.mllib.feature.{StandardScaler,StandardScalerModel}

object App {

  def main(args: Array[String]) = {

      val conf = new SparkConf().setAppName("App")
      val sc = new SparkContext(conf)
      val sqlContext = new SQLContext(sc)
      
      val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load("dataall.csv")
      df.show()

      df.printSchema()//get structure of the data

      val rows = df.rdd
      val double_rdd = rows.map(row=>{
            var i:Int = 3
            var result = Array(row.getDouble(2))
            while(i < 758){
                  if(i == 82){
                     result = result :+ row.getInt(i).toDouble  
                  }
                  else{
                       result = result :+ row.getDouble(i)
                  } 
                   i = i + 1 
            }
            result
      })//To get RDD[Vector]
      val aggre = double_rdd.map{row => Vectors.dense(row)}

      //Standardlizae Data
      val scaler = new StandardScaler(withMean = true, withStd = true).fit(aggre)
      val scaledData = aggre.map(v => scaler.transform(v))

      //Building Matrix
      val matrix = new RowMatrix(scaledData)
      val r = matrix.numRows
      val c = matrix.numCols
      println(r,c)//Print the size of Data

      //Compute Corvariance Matrix
      val cor: Matrix = matrix.computeCovariance()
      val rows_cor = matrixToRDD(cor)
      val mat_cor = new RowMatrix(rows_cor)
      val cor_rdd = mat_cor.rows.map(x => x.toArray.mkString(","))
      cor_rdd.saveAsTextFile("Corvariance.csv")//Save as csv

      //Get first 24 principal components
      val pc: Matrix = matrix.computePrincipalComponents(24)
      val projected = matrix.multiply(pc)//Get projected data
      val projected_rdd = projected.rows.map(x => x.toArray.mkString(","))
      projected_rdd.saveAsTextFile("ScaledProjectedData.csv")//Save as csv
      
      }
}
