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

package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl.nn.{SpatialConvolutionMap, SpatialFullConvolutionMap}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class SpatialFullConvolutionMapSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A SpatialFullConvolutionMap" should "generate correct output" in {
    val seed = 100
    RNG.setSeed(seed)

    val nfeat = 16
    val kW = 5
    val kH = 5
    val layer = new SpatialFullConvolutionMap[Double](
      SpatialConvolutionMap.oneToOne[Double](nfeat), kW, kH)

    Random.setSeed(seed)
    val input = Tensor[Double](16, 32, 32).apply1(e => Random.nextDouble())

    val code = "torch.manualSeed(" + seed + ")\n" +
      "layer = nn.SpatialFullConvolutionMap(nn.tables.oneToOne(16), 5, 5)\n" +
      "weight = layer.weight\n" +
      "bias = layer.bias \n" +
      "output = layer:forward(input) "

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("weight", "bias", "output"))

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]

    layer.weight.copy(luaWeight)
    layer.bias.copy(luaBias)
    val output = layer.updateOutput(input)

    val weight = layer.weight
    val bias = layer.bias
    weight should be equals luaWeight
    bias should be equals luaBias
    output should be equals luaOutput
  }

}
