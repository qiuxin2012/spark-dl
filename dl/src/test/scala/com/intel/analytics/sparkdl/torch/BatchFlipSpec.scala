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
package com.intel.analytics.sparkdl.torch

import com.intel.analytics.sparkdl.nn.{BatchFlip}
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


class BatchFlipSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A BatchFlip " should "generate correct output" in {
    val seed = 100
    RNG.setSeed(seed)
    var i = 0
    val input = Tensor[Float](5, 3, 32, 32).apply1(_ => {i += 1; i})

    val module = new BatchFlip[Float]()

    val code = "torch.manualSeed(" + seed + ")\n" +
      """require 'image'
        BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

        function BatchFlip:__init()
          parent.__init(self)
          self.train = true
        end

        function BatchFlip:updateOutput(input)
          if self.train then
            local bs = input:size(1)
            local flip_mask = torch.randperm(bs):le(bs/2)
            for i=1,input:size(1) do
              if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
            end
          end
          self.output:set(input)
          return self.output
        end
        module = nn.BatchFlip():float()
        output = module:forward(input)"""

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Float]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Float]]

    val start = System.nanoTime()
    val output = module.forward(input)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be(output)

  }
}
