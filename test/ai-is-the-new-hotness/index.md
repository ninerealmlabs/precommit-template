---
title: AI Is the New Hotness 🌎🔥
date: 2024-05-20
authors:
  - name: ahgraber
    link: https://github.com/ahgraber
    image: https://github.com/ahgraber.png
tags:
  # meta
  - opinion
  # ai/ml
  - generative AI
  - LLMs
series: []
layout: single
toc: true
math: true
draft: false
---

Meta provided insight into some of the costs of training LLMs in their Llama 2[^llama2] and Llama 3[^llama3] papers,
listing GPU hours and GPU power draw required to train the models, and **t\(CO_{2}\)eq**, or metric tons of Carbon
Dioxide Equivalent emitted. Of note, we see a huge generational increase from Llama2 to Llama3. It took 12x the
electricity to train Llama3-8B as it did to train Llama2-7B, and 6.5x the electricity to train Llama3-70B as it did its
prior generation. It is also possible to glean some insights from larger, closed models; OpenAI reported FLOPs and GPUs
in the GPT-3 paper[^gpt3], from which power consumption was estimated.[^gpt3-est]

{{< table path="power_emissions.csv" header="true" caption="Calculating t\(CO_{2}\)eq" >}}

The generational difference in GPU power draw (W) is due to upgrades in training hardware. GPT3 trained on NVIDIA
_V100_ accelerators (300-330W), Llama 2 trained on NVIDIA _A100_-80GB accelerators (TDP of 350 or 400W), and Llama 3
trained on NVIDIA _H100_-80GB accelerators (TDP of 700W).

> [!NOTE]
> Sidebar: `TDP` stands for "Thermal Design Power", and represents the maximum power draw of
> the component (a GPU, in this case) due to its ability to remain at a safe operating temperature.
>
> If you're familiar with overclocking, you've brushed against the concept of TDP.\
> At risk of oversimplifying - the goal of overclocking is to run a computer (CPU or GPU) faster than stock settings. This
> is possible because chips are limited to stay in a safe thermal envelope for all chips in all situations. These limits are
> typically done by regulating the amount of electricity a chip can draw; running faster requires more power, but more power
> generates more heat. The secret to overclocking, then, is to have a better-than-standard cooling configuration that can
> keep the chip within its safe thermal envelope, even at extreme power draw.

## Training is only half of the lifecycle

The power consumption reported by Meta applies solely to training the models; Meta has provided no information on
energy use during LLM inference. However, the 2023 research paper
["Power Hungry Processing: ⚡️ Watts ⚡️ Driving the Cost of AI Deployment?"](https://arxiv.org/abs/2311.16863)
provides some insight into overall machine learning inference energy use, reporting that "according to AWS, the largest
global cloud provider, inference is estimated to make up 80 to 90% of total ML cloud computing demand, whereas a 2021
publication by Meta attributed approximately one-third of their internal end-to-end ML carbon footprint to model
inference, with the remainder produced by data management, storage, and training; similarly, a 2022 study from Google
attributed 60% of its ML energy use to inference, compared to 40% for training."[^watts] From this same paper, we can
extrapolate that statistical machine learning (i.e., regressions, decision trees, etc.) are much less energy intensive
than generative models; thus the training::inference relationship for LLMs may not align with these older, broader
attributions. Luccioni et al. also found that autoregressive text generation with a Llama2-7B-analogous LLM (BLOOMZ-7B)
required approximately 0.1 kWh per 1000 generations of 10 tokens.[^watts] Alternatively, OpenAI estimated in its GPT-3
paper that "generating 100 pages of content from a trained model can cost on the order of 0.4 kW-hr."[^gpt3]. At 500
words/page and assuming (per OpenAI[^tokens]) 500 words ≈ 683 tokens, then 100 pages ≈ 68,300 tokens. Where a Llama2-7B
equivalent uses 0.1 kWh to generate 10k tokens, GPT-3 generates only 17k tokens with the same power.

> [!WARNING]
> Note: GPT-3 at 175B having greater efficiency than a 7B class model seems strange, but
> these metrics are not truly comparable. The 7B estimate is based on 1000, 10-token generations (and therefore must also
> process 1000 inputs of unknown length), while the GPT-3 estimate is derived from an assumption-laden calculation based
> on data that gives no insight into the number of inferences or length of inputs required to generate text of that
> cumulative length.

Unfortunately, it is ~~difficult~~ impossible to calculate _actual_ cumulative inference costs of open models because
they can be deployed by anyone, anywhere there is sufficient hardware to run the inference. Given that 7B (or 8B) class
models can be quantized to run on consumer GPUs, there are many systems that meet that hardware requirement (i.e., most
modern PCs with a discrete graphics card).

> [!NOTE]
> Luccioni et al. use the cost-per-generation estimates to calculate the breakeven point
> where the cost of inference meets the cost of training.
>
> Replicating this exercise:\
> Given the generation cost of the Llama2-7B analogue (0.1 kWh / 1k 10-tokens inferences), Llama2-7B needs an estimated
> 526 million 10-token inferences and Llama3-8B would require approximately 8.75 billion 10-token inferences to be
> equivalent to use the same energy used in training. For GPT-3, generating 52 million max-length responses (2k tokens)
> would use approximately the same energy as used in training. These numbers should only be used as estimates, because
> the length of the input context will also effect compute requirements (and therefore power consumption); none of the
> references controlled for input length.

## Estimating the maximum

Although accurate estimation of the electricity used by models deployed for inference is impossible, it _is_ possible
to do some back-of-the-envelope math and estimate the maximum possible energy used to power the GPUs used for training
and inference.

Assumptions:

- Only GPUs are included in this analysis (they're the compute accelerator of choice for most organizations). This
  excludes Google models, as Google trains on its own TPU infrastructure (Google has not published TDP numbers for its
  recent TPUs).[^tpu]

- In 2024, most training and inference will use the most recent GPU available generation -- NVIDIA A100 and H100/800
  series.[^nvidia] NVIDIA's Blackwell chips will not be delivered until 2025.

- While AMD accelerators _can_ be used to train models[^viking], they are much less frequently used. As AMD have - at
  best - 6% of NVIDIA share in the datacenter[^antares] [^gpu-poors], I will exclude them from the analysis.

- NVIDIA's production capacity is constrained by TSMC's ability to produce the chips themselves, thus dramatic changes
  in supply rate are unlikely.

- It is unlikely that these datacenter cards will be used solely for LLM inference. I am also not estimating any power
  consumption from older/smaller datacenter GPUs (A40, L40 series) or from non-datacenter GPUs. The assumption is that
  any LLM use with _non-A100/H100_ GPUs will be made up for by _non-LLM_ use _with_ A100/H100 GPUs... which is probably
  an underestimate given the number of people experimenting on consumer-class GPUs.

Estimates:

- Estimates have NVIDIA selling ~550k H100s in in 2023[^toms1], and selling ~500k A100s and H100s in Q3 2023
  alone.[^toms2] If all of 2023 looked like Q3, then NVIDIA sold approximately 140k H100s and 360k A100s per quarter.
- Other estimates have NVIDIA producing up to 3.5 million H100s 2023-2024.[^gpu-poors]

Given the A100:H100 ratio in the above estimates, I extrapolate that by the end of 2024, 3.5 million H100s and up to 9
million A100s will be in circulation. These numbers are likely high, but this exercise designed to estimate a
theoretical maximum.

The cumulative annual power use of these GPUs is ~53 million MWh/yr.

$$
\begin{align*}
9,000,000 \text{ A100s} \times 400 \text{W} \times 8,765 \text{ H/yr} & = 31,554,000 \text{ MWh/yr} \\
3,500,000 \text{ H100s} \times 700 \text{W} \times 8,765 \text{ H/yr} & = 21,474,250 \text{ MWh/yr}
\end{align*}
$$

## The catch

The reported power usage estimates are based solely on _GPU_ energy demands -- this does not encompass the rest of the
energy draw from the rest of the components in the server, nor any other associated power utilization from the data
center (networking, cooling). Meta themselves acknowledge this saying in the Llama 2 technical paper:

> It is important to note that our calculations do not account for further power demands, such as those from
> interconnect or non-GPU server power consumption, nor from datacenter cooling systems.[^llama2]

These additional power demands - the CPUs on the GPU servers, powering the extremely high-throughput network equipment,
other servers that process the data pipelines for trillions of tokens, and storage servers that house those trillions
of tokens- are unrecorded. I don't know enough about these systems or their scales to be able to estimate power draws,
but I would find it hard to imagine that the power draw of CPUs, networking gear and hard drives exceeds that of the
GPUs themselves. I'll estimate an upper limit for these unmeasured consumers by doubling the total GPU power
consumption.

Also unrecorded are the cooling requirements for such systems. Long-running systems will reach a thermal equilibrium
based on the system thermal envelope (as given by TDP), and the data center's ability to move the heat away from the
system. Data center thermal management can be done with air conditioning and lots of fans, but air conditioning
requires electricity (as much or more than the equipment generating the heat; Watts of heat vs Watts of cooling +
losses from inefficiencies). For data centers that use air conditioning, the power consumption number must _again_ be
doubled to account for this power draw.

|                                                              | GPU-only Utilization (MWh) | + Datacenter (MWh) | + A/C (MWh) |
| -----------------------------------------------------------: | -------------------------: | -----------------: | ----------: |
|                                Llama 3 8B<br>(training only) |                        910 |              1,820 |       3,640 |
|                               Llama 3 70B<br>(training only) |                      4,480 |              8,960 |      17,920 |
| theoretical GPU max consumption<br>Training and/or Inference |                 53,000,000 |        106,000,000 | 212,000,000 |

Total global electricity production is estimated to be 24,816,400 GWh/year.[^wiki] The theoretical all-GPU max
consumption (including datacenter additions) is 106 GWh/year, or 0.0004% of the global production.

## \(CO_{2}\) emissions

<!-- markdownlint-disable MD028 -->

> The actual power usage of a GPU is dependent on its utilization and is likely to vary from the Thermal Design Power
> (TDP) that we employ as an estimation for GPU power... We estimate the total emissions for training to be 539
> t\(CO_{2}\)eq, of which 100% were directly offset by Meta's sustainability program.[^llama2]

> Estimated total emissions were 2290 t\(CO_{2}\)eq, 100% of which were offset by Meta's sustainability
> program.[^llama3]

<!-- markdownlint-enable MD028 -->

As mentioned above, these numbers represent GPU use only, and do not account for the rest of the server or datacenter
infrastructure.

How did they ascertain t\(CO_{2}\)eq?

$$
\begin{align*}
\text{power consumed (kWh)} &= \frac{\text{GPU time (Hours)} \times \text{GPU power draw (W)}}{1000} \\
\text{metric tons CO$_2$ equivalent (tCO$_2$eq)} &= \text{power consumed (kWh)} \times \text{emission rate (CO$_2$/kWh)}
\end{align*}
$$

where the \(CO_{2}\) to kWh emission rate, `0.000417`, is given by the EPA based on the US national annual average
\(CO_{2}\) output rate in 2021.[^epa] While not exactly as claimed by Meta, this recalculation is fairly close!

|                                                              | GPU-only consumption (MWh) | GPU-only emissions<br>(t\(CO_{2}\)eq) | + Datacenter (MWh) | + Datacenter emissions<br>(t\(CO_{2}\)eq) |
| -----------------------------------------------------------: | -------------------------: | -----------------------------------: | -----------------: | ---------------------------------------: |
|                                Llama 3 8B<br>(training only) |                        910 |                                  380 |              1,820 |                                      760 |
|                               Llama 3 70B<br>(training only) |                      4,480 |                                 1870 |              8,960 |                                     3740 |
| theoretical GPU max consumption<br>Training and/or Inference |                 53,000,000 |                           22,000,000 |        106,000,000 |                               44,000,000 |

> [!NOTE]
> You may notice that I did not include the A/C power consumption in the \(CO_{2}\)
> calculation above. I'll get to that in the next section.

Extending our max-consumption exercise from above, the estimated \(CO_{2}\) emissions from running all recent
datacenter GPUs and their associated hardware is 44 million metric tons per year. Or, given the estimates from "Power
Hungry Processing", each 1,000 10-token inferences releases 0.0000417 t\(CO_{2}\)eq, or about (1/10^{th}) of a pound of
\(CO_{2}\).

To put these numbers in scale, I'll share some equivalents I've found using the EPA's equivalencies calculator[^epa]:

|                                                              |        GPU-only emissions equivalent        |             + Datacenter emissions equivalent             |
| -----------------------------------------------------------: | :-----------------------------------------: | :-------------------------------------------------------: |
|                                    GPT-3<br> (training only) |   3 SF-NYC round-trip flights [^gpt3-est]   |            1 year of electricity for 212 homes            |
|                                Llama 3 8B<br>(training only) |     1 year of electricity for 75 homes      | 2 million miles driven by a gas-powered passenger vehicle |
|                               Llama 3 70B<br>(training only) |     45 tanker trucks' worth of gasoline     |            1 year of electricity for 740 homes            |
| theoretical GPU max consumption<br>Training and/or Inference | 1 year of electricity for 4.3 million homes |       11 coal-fired power plants running for 1 year       |
|                1,000 10-token inferences<br>(7B-class model) |           2.8 smartphone charges            |     0.21 mi driven by a gas-powered passenger vehicle     |

The world's largest carbon capture facility can capture 36,000 tons of \(CO_{2}\) per year.[^capture] Capturing the
carbon emitted by the electricity consumed by datacenters hosting GPU clusters would require 1,200 of them.

> [!NOTE]
> Microsoft just published their [2024 Environmental Sustainability Report](https://query.prod.cms.rt.microsoft.com/cms/api/am/binary/RW1lhhu). In it,
> they note that while their direct \(CO_{2}\) emissions from the datacenter have actually decreased from 2020, their
> overall \(CO_{2}\) emissions have _increased_, due to emissions related to constructing new data centers.

## Water consumption

Reviewing from above -- a GPU's performance is constrained by TDP, the relationship between the power it draws, the
heat it generates from that power, and the cooling it receives. If a GPU is not cooled sufficiently, it may
automatically _thermal throttle_, or reduce its power draw (and thus its performance) in order to reduce its
temperature (or rather, maintain a safe operating temperature). Sufficient cooling is critical for ensuring computers
run at their optimal capacity. Cooling can be provided by (electrically-powered) air conditioning, but as discussed
above, it requires at least as much power consumption to cool as it does to run the equipment itself (heating joules ≤
cooling joules).

Non-refrigerated cooling methods are possible; most use evaporative cooling. From an electricity and emissions
standpoint, this is extremely efficient. Evaporation is used to either cool the air that goes into the data center
(like a swamp cooler), or to cool the hot-side heat exchanger that allows the data center' to move the heat away from
the servers.[^thirsty] This leads to enormous water consumption, a problem acknowledged by providers like Microsoft,
OpenAI, and Google.[^cooling] Li, et al. report:

> Training GPT3 in Microsoft's state-of-the-art U.S. data centers can consume a total of 5.4 million liters of water,
> including 700,000 liters of scope-1 on-site water consumption. Additionally, GPT-3 needs to "drink" (i.e., consume) a
> 500ml bottle of water for roughly 10-50 responses, depending on when and where it is deployed. These numbers may
> increase for the newly-launched GPT-4 that reportedly has a substantially larger model size.[^thirsty]

Microsoft shared in their 2024 sustainability report that they consumed over 8 million megaliters (over 2 billion gallons) of water in 2023, but did not break consumption down by source.[^microsoft-2024-sust] Although "consumption" here does
not "lose" water per se (the water re-enters the water cycle), the evaporation of water removes it from the local area,
which has negative impacts on "high stress" areas effected by drought or extreme temperatures.

There's one other method that uses water to cool computers -- direct radiation. Microsoft has experimented with sinking
data centers into the ocean and using water temperatures to cool the server components.[^underwater] Given concerns
about ocean warming and icecap melting, I wanted to understand how much directly dumping heat into the ocean would
effect temperatures:\
Total global electricity production is estimated to be 24,816,400 GWh/year.[^wiki] If the global electricity production
was used directly to heat the ocean with no losses, it would increase global ocean temperature by 0.00002\(\degree\)C per
year, assuming electricity production does not change. What a relief!(?) Of course, this assumes the heat is
dissapated; I imagine that exposing 90\(\degree\)C hotspots directly to ocean ecosystems would have negative impacts.

<!-- ```sh
#   global power production / ocean volume / water density / water specific heat
units -t '24816400 GWh / (1347000000 km^3) / waterdensity / water_specificheat'
``` -->

## AI has environmental impacts, regardless of offsets

In both their Llama 2 and LLama 3 papers, Meta shared the t\(CO_{2}\) emissions equivalents from the electricity use
for training the models and trumpeted that "100% [of the \(CO_{2}\) emissions] were directly offset by Meta's
sustainability program"[^llama2]. Microsoft promises, "by 2030, 100% of our electricity consumption will be matched by
zero carbon energy purchases 100% of the time"[^microsoft-2024-sust]. Further, companies like Google, Meta, and
Microsoft are incredibly aware of the optics of their energy use, and publish sustainability reports and fund green
energy generation, carbon offset and reclamation, and water replenishment projects. While the disclosures (and the
sustainability efforts) are admirable, the premise that these models are therefore emissions neutral because of
emissions offsets or using only sustainably-sourced electricity are flawed.

Electricity consumption and carbon emissions are a zero-sum game. Renewable-based electricity used for Generative AI
could have been used elsewhere had it not been used for AI. Carbon credits or carbon capture used to offset the
electricity emissions used for Generative AI could have been used elsewhere had it not been used for AI. By using
energy to train (or inference) a model, that electricity is removed from circulation and we accept the opportunity cost
of using that energy elsewhere. Until _all electricity_ is emissions-free, it is disingenuous to imply that the
consumption is emissions-free.

## Conclusion & some soapboxing

Use of AI has direct environmental impacts from the use of electricity, water consumption, and embodied emissions
(which are beyond the scope of this post). As Joshua (the AI in the 1983 classic movie War Games) learned, "the only
winning move is not to play"; the best way to minimize environmental impact of Generative AI is to not train models or
use them for inference at all. As an AI/ML Engineer, I understand that this is not a viable solution for most
businesses (or my job security).

So, what can we do?

As a practitioner:

- Use the smallest model that accomplishes your task. By "smallest", I mean in terms of model size (use Phi 3 3.8B
  instead of GPT-4 if it is sufficient for your needs), but also in terms of model selection (don't use an LLM to
  classify text if a small embedding model and logistic regression works).
- Efficiently organize training and inference tasks toward optimizing utilization.
- Consider scheduling workloads with respect to energy efficiency or renewable energy availability. (If power is
  provided via solar panels, running tasks at night doesn't use renewable energy.)

As a corporation:

- Efficiently deploy infrastructure to minimize underutilization.
- Ensure deployed infrastructure is configured to do dynamic allocation and scaling (spin up when needed; down when not
  in use).
- Select datacenters or deployment zones based on access to greener energy and/or better water availability. (Despite
  what I say above about the zero-sum game, it's still preferable to deploy to locations where the environmental cost
  is minimized.)

As a hyperscaler/provider:

- Continue funding renewable energy, carbon reclamation, and water replenishment projects.
- Provide further transparency on per-datacenter energy and water consumption to aid data center selection.
- Provide granular reporting on costs of training and projected costs of inference.

## Updates

Meta's 2024 sustainability report shares that data center electricity use increased 34% from 2023.
Data center water consumption increased 7%, though groundwater consumption increased 137% and consumption from areas with high or extremely high baseline water stress increased 20%.
Meta continues to build out its datacenter infrastructure. [^meta-consumption-increase]

## Works Cited

[^llama2]: [[2307.09288] Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

[^llama3]: [Llama3 | MODEL_CARD](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)

[^gpt3]: [[2005.14165] Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)

[^gpt3-est]: [[2104.10350] Carbon Emissions and Large Neural Network Training](https://arxiv.org/pdf/2104.10350)

[^watts]: [[2311.16863] Power Hungry Processing: Watts Driving the Cost of AI Deployment?](https://arxiv.org/abs/2311.16863)

[^tokens]: [What are tokens and how to count them? | OpenAI Help Center](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)

[^tpu]: [Google Tensor Processing Unit](https://en.wikipedia.org/wiki/Tensor_Processing_Unit#Fifth_generation_TPU)

[^nvidia]: [Nvidia datacenter GPUs](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units#Data_Center_GPUs)

[^viking]: [Viking 7B: The first open LLM for the Nordic languages](https://www.silo.ai//blog/viking-7b-the-first-open-llm-for-the-nordic-languages)

[^antares]: [How The "Antares" MI300 GPU Ramp Will Save AMD's Datacenter Business](https://www.nextplatform.com/2024/01/31/how-the-antares-mi300-gpu-ramp-will-save-amds-datacenter-business/)

[^gpu-poors]: [Google Gemini Eats The World – Gemini Smashes GPT-4 By 5X, The GPU-Poors](https://www.semianalysis.com/p/google-gemini-eats-the-world-gemini)

[^toms1]: [Nvidia to Sell 550,000 H100 GPUs for AI in 2023: Report | Tom's Hardware](https://www.tomshardware.com/news/nvidia-to-sell-550000-h100-compute-gpus-in-2023-report)

[^toms2]: [Nvidia sold half a million H100 AI GPUs in Q3 thanks to Meta, Facebook — lead times stretch up to 52 weeks: Report | Tom's Hardware](https://www.tomshardware.com/tech-industry/nvidia-ai-and-hpc-gpu-sales-reportedly-approached-half-a-million-units-in-q3-thanks-to-meta-facebook)

[^wiki]: [List of countries by electricity production](https://en.wikipedia.org/wiki/List_of_countries_by_electricity_production)

[^epa]: [Greenhouse Gases Equivalencies Calculator - Calculations and References | US EPA](https://www.epa.gov/energy/greenhouse-gases-equivalencies-calculator-calculations-and-references)

[^capture]: [The world's largest direct carbon capture plant just went online](https://www.engadget.com/the-worlds-largest-direct-carbon-capture-plant-just-went-online-172447811.html)

[^thirsty]: [[2304.03271] Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models](https://arxiv.org/abs/2304.03271)

[^cooling]: [Artificial intelligence technology behind ChatGPT was built in Iowa — with a lot of water | AP News](https://apnews.com/article/chatgpt-gpt4-iowa-ai-water-consumption-microsoft-f551fde98083d17a7e8d904f8be822c4)

[^microsoft-2024-sust]: [Our 2024 Environmental Sustainability Report - Microsoft On the Issues](https://blogs.microsoft.com/on-the-issues/2024/05/15/microsoft-environmental-sustainability-report-2024/)

[^underwater]: [Microsoft finds underwater datacenters are reliable, practical and use energy sustainably - Source](https://news.microsoft.com/source/features/sustainability/project-natick-underwater-datacenter/)

[^meta-consumption-increase]: [Meta data center electricity consumption hits 14,975GWh, leased data center use nearly doubles - DCD](https://www.datacenterdynamics.com/en/news/meta-data-center-electricity-consumption-hits-14975gwh-leased-data-center-use-nearly-doubles/)
