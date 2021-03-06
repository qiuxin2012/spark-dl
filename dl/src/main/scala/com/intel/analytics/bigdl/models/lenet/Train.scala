/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.models.lenet

import java.nio.file.Paths

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.{GreyImgNormalizer, GreyImgToBatch, SampleToGreyImg}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext


object Train {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Utils._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val sc = Engine.init(param.nodeNumber, param.coreNumber, param.env == "spark").map(conf => {
        conf.setAppName("Train Lenet on MNIST")
          .set("spark.akka.frameSize", 64.toString)
          .set("spark.task.maxFailures", "1")
        new SparkContext(conf)
      })

      val trainData = Paths.get(param.folder, "/train-images-idx3-ubyte")
      val trainLabel = Paths.get(param.folder, "/train-labels-idx1-ubyte")
      val validationData = Paths.get(param.folder, "/t10k-images-idx3-ubyte")
      val validationLabel = Paths.get(param.folder, "/t10k-labels-idx1-ubyte")

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        LeNet5(classNum = 10)
      }

      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T(
          "learningRate" -> param.learningRate
        )
      }

      val trainSet = (if (sc.isDefined) {
        DataSet.array(load(trainData, trainLabel), sc.get)
      } else {
        DataSet.array(load(trainData, trainLabel))
      }) -> SampleToGreyImg(28, 28) -> GreyImgNormalizer(trainMean, trainStd) -> GreyImgToBatch(
        param.batchSize)

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = ClassNLLCriterion[Float]())
      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }

      val validationSet = (if (sc.isDefined) {
        DataSet.array(load(validationData, validationLabel), sc.get)
      } else {
        DataSet.array(load(validationData, validationLabel))
      }) -> SampleToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToBatch(
        param.batchSize)

      optimizer
        .setValidation(
          trigger = Trigger.everyEpoch,
          dataset = validationSet,
          vMethods = Array(new Top1Accuracy))
        .setState(state)
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
        .optimize()
    })
  }
}
