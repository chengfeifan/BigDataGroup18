package com.spark.demo1

import java.util.Calendar
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.{SQLContext,Row}
import com.databricks.spark.csv._
import org.apache.spark.mllib.rdd
import org.apache.spark.rdd
import org.apache.spark.Logging
import org.apache.spark.graphx.{Edge, Graph, _}
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import scala.collection.mutable.ArrayBuffer
import Array._
import scala.io.Source


class ANN private (private val trainingRDD: RDD[(Vector, Vector)],
                  private val hiddenLayersTopology: Array[Int],
                  private var maxIterations: Int) extends Serializable with Logging{
 private val layersCount = hiddenLayersTopology.length + 2 //input and output layers included
 private val MaxLayerCount = 1e10.toLong
 private def getVertexId(layer: Int, index:Int): Long ={
   return layer * MaxLayerCount + index
 }
 private var graph = {
   val topology = 0 +: convertTopology(trainingRDD, hiddenLayersTopology)
   val vertices =
     (1 to layersCount - 1).flatMap( layerIndex =>{
       val layerLength = topology(layerIndex)
       (0 to layerLength).map( i => {
         val vertexId = getVertexId(layerIndex, i)
         (vertexId, (layerIndex, i, 1.0, 0.0)) //layer, index, value, delta
       })
     })
   .union((1 to topology(layersCount)).map( i => {
       val vertexId = getVertexId(layersCount, i)
       (vertexId, (layersCount, i, 1.0, 0.0))
     })) //last layer without bias
   val edges = (2 to layersCount).flatMap(layerIndex =>{
     val preLayer = layerIndex - 1
     val prelayerLength = topology(layerIndex - 1)
     val layerLength = topology(layerIndex)
     val buffer = new ArrayBuffer[Edge[Double]]()
     for(target <- 1 to layerLength)
       for(src <- 0 to prelayerLength){
         val srcId = getVertexId(preLayer, src)
         val targetId = getVertexId(layerIndex, target)
         buffer += Edge(srcId, targetId, scala.util.Random.nextDouble())
       }
     buffer
   })
   val verticesRdd: RDD[(VertexId, (Int, Int, Double, Double))] = trainingRDD.context.parallelize(vertices)
   val edgesRdd: RDD[Edge[Double]] = trainingRDD.context.parallelize(edges)
   Graph(verticesRdd, edgesRdd).partitionBy(PartitionStrategy.CanonicalRandomVertexCut)
 }
 var forwardCount = graph.vertices.sparkContext.accumulator(0)
 private def convertTopology(input: RDD[(Vector,Vector)],
                             hiddenLayersTopology: Array[Int] ): Array[Int] = {
   val firstElt = input.first
   firstElt._1.size +: hiddenLayersTopology :+ firstElt._2.size
 }
 def run(trainingRDD: RDD[(Vector, Vector)]): Unit ={
   var i = 1
   val data = trainingRDD.collect()
   while(i < maxIterations){
     var diff = 0.0
     data.foreach(sample => {
       val d = this.Epoch(sample._1, sample._2)
       diff += d
       println(d)
     })
     logError(s"iteration $i get " + diff)
     i += 1
   }
   println("forwardCount: " + forwardCount.value)
 }
 def Epoch(input: Vector, output: Vector): Double = {
   val preGraph = graph
   graph.cache()
   assignInput(input)
   for(i <- 1 to (layersCount - 1)){
     forward(i)
   }
   val diff = ComputeDeltaForLastLayer(output)
    for(i <- layersCount until 2 by -1){
      backporgation(i)
    }
    updateWeights()
    preGraph.unpersist()
    diff
  }

  private def assignInput(input: Vector): Unit = {

    val in = graph.vertices.context.parallelize(input.toArray.zipWithIndex)
    val inputRDD = in.map( x => (x._2 + 1, x._1)).map(x =>{
      val id = getVertexId(1, x._1)
      (id, x._2)
    })

    graph = graph.joinVertices(inputRDD){
      (id, oldVal, input) => (oldVal._1, oldVal._2, input, oldVal._4)
    }
  }

  /**
   * Feed forward from layerIndex to layerIndex + 1
   */
  private def forward(layerIndex: Int):Unit ={

    val sumRdd:VertexRDD[Double] = graph.subgraph(edge => edge.srcAttr._1 == layerIndex).aggregateMessages[Double](
      triplet => {
        val value = triplet.srcAttr._3
        val weight = triplet.attr
        triplet.sendToDst(value * weight)
      }, _ + _, TripletFields.Src
    )
    forwardCount += 1

    graph = graph.joinVertices(sumRdd){
      (id, oldRank, msgSum) => (oldRank._1, oldRank._2, breeze.numerics.sigmoid(msgSum), oldRank._4)
    }

  }

  /**
   * from layerIndex to layerIndex - 1
   */
  private def backporgation(layerIndex: Int): Unit ={
    val deltaRdd = graph.subgraph(edge => edge.dstAttr._1 == layerIndex).aggregateMessages[Double](
      triplet => {
        val delta = triplet.dstAttr._4
        val weight = triplet.attr
        triplet.sendToSrc(delta * weight)
      }, _ + _, TripletFields.Dst
    )

    graph = graph.joinVertices(deltaRdd){
      (id, oldValue, deltaSum) => {
        val e = deltaSum
        (oldValue._1, oldValue._2, oldValue._3, e)     // update delta
      }
    }
  }

  private def updateWeights(): Unit ={
    val eta = 10

    graph = graph.mapTriplets(triplet =>{
      val delta = triplet.dstAttr._4
      val y = triplet.dstAttr._3
      val x = triplet.srcAttr._3
      val newWeight = triplet.attr + eta * delta * y * (1.0 - y) * x
      newWeight
    })
  }

  private def ComputeDeltaForLastLayer(output: Vector): Double ={
    var sampleDelta = graph.vertices.sparkContext.accumulator(0.0)
    graph = graph.mapVertices( (id, attr) => {
      if(attr._1 != layersCount){
        attr
      }
      else{
        val index = attr._2
        val d = output(index - 1)
        val y = attr._3
        val delta = (d - y)
        sampleDelta += (d - y) * (d - y)
        (attr._1, attr._2, attr._3, delta)
      }
    })
    graph.vertices.count()
    sampleDelta.value * 0.5
  }

  def predict(input: Vector): Vector ={
    assignInput(input)
    for(i <- 1 to (layersCount - 1)){
      forward(i)
    }
    val result = graph.vertices.filter( x=> x._2._1 == layersCount).map(x => x._2).map(x => (x._2, x._3))
    val doubles = result.sortBy(x => x._1).map(x => x._2).collect()
    Vectors.dense(doubles)
  }
}


/**
 * Top level methods for training the artificial neural network (ANN)
 */
object ANN {

  private val defaultTolerance: Double = 1e-4

  def train(trainingRDD: RDD[(Vector, Vector)],
            hiddenLayersTopology: Array[Int],
            maxNumIterations: Int) : ANN = {

    val ann = new ANN(trainingRDD, hiddenLayersTopology, maxNumIterations)
    ann.run(trainingRDD)
    return ann
  }

}






object App{

  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = new SparkConf().setAppName("Parallel ANN").setMaster("local[4]")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("/NeuralNetwork/output")

//SQL => Vector		
// Load and parse the data

	var arr = new scala.collection.mutable.ArrayBuffer[(Vector, Vector)]()
	

	var line = Source.fromFile("/root/ANNScaledProjectedData.txt").getLines.next()
	var parts = line.split('\t')

	for(i <- 0 to 4999){
		var tmpTrain = Vector(Vectors.dense(parts(i+60).toDouble), Vectors.dense(parts(0+i).toDouble,parts(1+i).toDouble,parts(2+i).toDouble,parts(3+i).toDouble,parts(4+i).toDouble,parts(5+i).toDouble,parts(6+i).toDouble,parts(7+i).toDouble,parts(8+i).toDouble,parts(9+i).toDouble,parts(10+i).toDouble,parts(11+i).toDouble,parts(12+i).toDouble,parts(13+i).toDouble,parts(14+i).toDouble,parts(15+i).toDouble,parts(16+i).toDouble,parts(17+i).toDouble,parts(18+i).toDouble,parts(19+i).toDouble,parts(20+i).toDouble,parts(21+i).toDouble,parts(22+i).toDouble,parts(23+i).toDouble,parts(24+i).toDouble,parts(25+i).toDouble,parts(26+i).toDouble,parts(27+i).toDouble,parts(28+i).toDouble,parts(29+i).toDouble,parts(30+i).toDouble,parts(31+i).toDouble,parts(31+i).toDouble,parts(32+i).toDouble,parts(33+i).toDouble,parts(34+i).toDouble,parts(35+i).toDouble,parts(36+i).toDouble,parts(37+i).toDouble,parts(38+i).toDouble,parts(39+i).toDouble,parts(40+i).toDouble))
		arr += new Tuple2(tmpTrain(1), tmpTrain(0))		
	}
	
	
	
//Time
    val startTime = System.nanoTime()
    val ann = ANN.train(sc.parallelize(arr), Array(2), 100)
    val elapsed = (System.nanoTime() - startTime) / 1e9
	
    println(s"Finished training NN model.  Summary:")
    println(s"\t Training time: $elapsed sec")
	
//Result of ANN	
	for(j<-5000 to 10000){	
		var tmpTest = Vectors.dense(parts(0+j).toDouble,parts(1+j).toDouble,parts(2+j).toDouble,parts(3+j).toDouble,parts(4+j).toDouble,parts(5+j).toDouble,parts(6+j).toDouble,parts(7+j).toDouble,parts(8+j).toDouble,parts(9+j).toDouble,parts(10+j).toDouble,parts(11+j).toDouble,parts(12+j).toDouble,parts(13+j).toDouble,parts(14+j).toDouble,parts(15+j).toDouble,parts(16+j).toDouble,parts(17+j).toDouble,parts(18+j).toDouble,parts(19+j).toDouble,parts(20+j).toDouble,parts(21+j).toDouble,parts(22+j).toDouble,parts(23+j).toDouble,parts(24+j).toDouble,parts(25+j).toDouble,parts(26+j).toDouble,parts(27+j).toDouble,parts(28+j).toDouble,parts(29+j).toDouble,parts(30+j).toDouble,parts(31+j).toDouble,parts(31+j).toDouble,parts(32+j).toDouble,parts(33+j).toDouble,parts(34+j).toDouble,parts(35+j).toDouble,parts(36+j).toDouble,parts(37+j).toDouble,parts(38+j).toDouble,parts(39+j).toDouble,parts(40+j).toDouble)
		var pre = ann.predict(tmpTest)
		println(pre)
	}	
	
    sc.stop()
  }
}
