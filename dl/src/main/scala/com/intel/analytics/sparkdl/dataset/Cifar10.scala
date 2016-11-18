/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.sparkdl.dataset

import java.nio.file.{Files, Path, Paths}

import com.intel.analytics.sparkdl.models.cifar.VggLike
import com.intel.analytics.sparkdl.nn.ClassNLLCriterion
import com.intel.analytics.sparkdl.optim.SGD.{EpochStep, EpochStep2}
import com.intel.analytics.sparkdl.optim.{LocalOptimizer, SGD, Top1Accuracy, Trigger}
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.T
import scopt.OptionParser

object Cifar10 {
  case class Cifar10LocalParam(
    folder: String = "./",
    net: String = "vgg"
  )

  private val parser = new OptionParser[Cifar10LocalParam]("Spark-DL Cifar10 Local Example") {
    head("Spark-DL Cifar10 Local Example")
    opt[String]('f', "folder")
      .text("where you put the Cifar10 data")
      .action((x, c) => c.copy(folder = x))
  }

  def main(args: Array[String]) {
    parser.parse(args, new Cifar10LocalParam()).map(param => {
      val trainDataSource = new CifarDataSource(Paths.get(param.folder + "/train"), looped = true)
      val validationDataSource = new CifarDataSource(Paths.get(param.folder + "/val"),
        looped = false)
      val arrayToImage = ArrayByteToRGBImage(1)
      val toTensor = new RGBImageToTensor(batchSize = 128)
      val bgr2yuv = BgrToYuv()
      val normalizer = YuvImageNormalizer(trainDataSource -> arrayToImage -> toTensor -> bgr2yuv)
      val validationNormalizer = YuvImageNormalizer(validationDataSource -> arrayToImage -> toTensor -> bgr2yuv)
      val model = VggLike[Float](classNum = 10)
      println(model)

      val optimizer = new LocalOptimizer[Float](
        data = trainDataSource -> arrayToImage -> toTensor -> bgr2yuv -> normalizer,
        validationData = validationDataSource -> arrayToImage -> toTensor -> bgr2yuv -> validationNormalizer,
        model,
        criterion = new ClassNLLCriterion[Float](),
        optimMethod = new SGD[Float](),
        state = T(
          "learningRate" -> 1.0,
          "learningRateDecay" -> 1e-7,
          "weightDecay" -> 0.0005,
          "momentum" -> 0.9,
          "dampening" -> 0.0,
          "learningRateSchedule" -> EpochStep2(25, 0.5)
        ),
        endWhen = Trigger.maxEpoch(250)
      )
      optimizer.setValidationTrigger(Trigger.everyEpoch)
      optimizer.addValidation(new Top1Accuracy[Float])

      optimizer.optimize()
    })
  }
}
