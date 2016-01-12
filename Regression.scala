package com.spark.demo1

import Array._
import scala.collection.mutable.ArrayBuffer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LassoModel
import org.apache.spark.mllib.regression.LassoWithSGD
/**
 * Created by mymac on 15/11/3.
 */
object App {

  def main(args: Array[String]) = {

      val conf = new SparkConf().setAppName("App")
        val sc = new SparkContext(conf)

// Load and parse the data
val data = sc.textFile("Datatest9564.txt")
val modelarr:Array[LinearRegressionModel] = new Array[LinearRegressionModel](24)
//val errarr:Array[Double] = new Array[Double](24)

for(i <- 0 to 23) {
val parsedData = data.map { line =>
  val parts = line.split(',')
  val parts0 = parts(0).split('\t')
  val parts1 = parts(1).split('\t')
  val parts2 = parts(2).split('\t')
  LabeledPoint(parts2(i).toDouble, Vectors.dense(concat(parts0,parts1).map(_.toDouble)))
}.cache()

// Building the model
val numIterations = 1000
val stepSize = 0.01
val miniBatchFraction = 1.0
val initialWeights:Array[Double] = Array(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
initialWeights(i) = -1
initialWeights(24 + i) = 2
val model = LinearRegressionWithSGD.train(parsedData, numIterations, stepSize, miniBatchFraction, Vectors.dense(initialWeights))

// Evaluate model on training examples and compute training error
//val valuesAndPreds = parsedData.map { point =>
//  val prediction = model.predict(point.features)
//  (point.label, prediction)
//}
//val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
//println("training Mean Squared Error = " + MSE)

//errarr(i) = MSE
modelarr(i) = model
}

//for(i <- 0 to 23) {
//println(errarr(i))
//}

for(j <- 0 to 23) {
for(i <- 0 to 47) {
val inputtest:Array[Double] = Array(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)// = Array(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
inputtest(i) = 1
print(modelarr(j).predict(Vectors.dense(inputtest)))
print(' ')
}
println()
println()
}

//model.save(sc, "Modeltest")
//val sameModel = LinearRegressionModel.load(sc, "Model0")
      }
}
