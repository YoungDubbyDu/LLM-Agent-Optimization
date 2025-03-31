# LLM-Agent-Optimization
This is the reading list for the survey "A Survey of LLM-based Agents Optimization", which systematically explores optimization techniques for enhancing LLM-based agents. The survey categorizes existing works into parameter-driven optimization, parameter-free optimization, datasets and benchmarks, and real-world applications. We will keep adding papers and improving the list. Any suggestions and PRs are welcome!


<div align="center">
  <img src="https://github.com/user-attachments/assets/7ad2d1e2-17c7-42bc-bcbc-a615209b1a5a" width="50%">
</div>


# Parameter-driven Optimization

## Conventional Fine-Tuning-based

- FireAct : TOWARD LANGUAGE AGENT FINE-TUNING  (arXiv 2023) [[paper](https://arxiv.org/pdf/2310.05915)] [[code](https://github.com/anchen1011/FireAct)]
- AgentTuning: Enabling Generalized Agent Abilities for LLMs  (ACL-findings 2024) [[paper](https://arxiv.org/pdf/2310.12823)] [[code](https://github.com/THUDM/AgentTuning)]
- SMART: Synergistic Multi-Agent Framework with Trajectory Learning for Knowledge-Intensive Tasks (arXiv 2024) [[paper](https://arxiv.org/abs/2407.09893)] [[code](https://github.com/yueshengbin/SMART)]
- Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for Large Language Models (ACL-findings 2024) [[paper](https://arxiv.org/abs/2403.12881)] [[code](https://github.com/InternLM/Agent-FLAN)]
- Bootstrapping LLM-based Task-Oriented Dialogue Agents via Self-Talk (arXiv 2024) [[paper](https://arxiv.org/abs/2401.05033)]
- SaySelf: Teaching LLMs to Express Confidence with Self-Reflective Rationales (EMNLP 2024) [[paper](https://arxiv.org/abs/2405.20974)] [[code](https://github.com/tianyang-x/SaySelf)]
- AgentGym: Evolving Large Language Model-based Agents across Diverse Environments (arXiv 2024) [[paper](https://arxiv.org/abs/2406.04151)] [[code](https://github.com/WooooDyy/AgentGym)]
- Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents (ACL 2024) [[paper](https://aclanthology.org/2024.acl-long.409.pdf)] [[code](https://github.com/Yifan-Song793/ETO)]
- Agent LUMOS: Unified and Modular Training for Open-Source Language Agents (ACL 2024) [[paper](https://arxiv.org/pdf/2311.05657v3)] [[code](https://allenai.github.io/lumos/)]
- LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error (ACL 2024) [[paper](https://arxiv.org/abs/2403.04746)] [[code](https://github.com/microsoft/simulated-trial-and-error)]
- NAT: Learning From Failure: Integrating Negative Examples when Fine-tuning LLMs as Agents (arXiv 2024) [[paper](https://arxiv.org/abs/2402.11651)] [[code](https://github.com/Reason-Wang/NAT)]
- OPTIMA: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System (arXiv 2024) [[paper](https://arxiv.org/abs/2410.08115)] [[code](https://chenweize1998.github.io/optima-project-page/)]
- Enhancing the General Agent Capabilities of Low-Parameter LLMs through Tuning and Multi-Branch Reasoning (NAACL 2024) [[paper](https://aclanthology.org/2024.findings-naacl.184/)] [[code](https://github.com/HAIV-Lab/LLM-TMBR)]
- AgentOhana: Design Unified Data and Training Pipeline for Effective Agent Learning (arXiv 2024) [[paper](https://arxiv.org/abs/2402.15506)] [[code](https://github.com/SalesforceAIResearch/xLAM)]
- TORA: A TOOL-INTEGRATED REASONING AGENT FOR MATHEMATICAL PROBLEM SOLVING (ICLR 2024) [[paper](https://arxiv.org/abs/2309.17452)] [[code](https://github.com/microsoft/tora)]
- ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent (arxiv 2023) [[paper](https://arxiv.org/pdf/2312.10003)]
- AGENTBANK: Towards Generalized LLM Agents via Fine-Tuning on 50000+ Interaction Trajectories (ACL-Findings 2024) [[paper](https://arxiv.org/abs/2410.07706)] [[code](https://huggingface.co/datasets/Solaris99/AgentBank)]
- ADASWITCH: Adaptive Switching between Small and Large Agents for Effective Cloud-Local Collaborative Learning (EMNLP 2024) [[paper](https://arxiv.org/abs/2410.13181)]
- Watch Every Step! LLM Agent Learning via Iterative Step-Level Process Refinement (EMNLP 2024) [[paper](https://arxiv.org/abs/2406.11176)] [[code](https://github.com/WeiminXiong/IPR)]
- Re-ReST: Reflection-Reinforced Self-Training for Language Agents (EMNLP 2024) [[paper](https://arxiv.org/abs/2406.01495)] [[code](https://github.com/PlusLabNLP/Re-ReST)]
- Retrospex: Language Agent Meets Offline Reinforcement Learning Critic (EMNLP 2024) [[paper](https://aclanthology.org/2024.emnlp-main.268/)] [[code](https://github.com/Yufei-Xiang/Retrospex)]
- ATM: Adversarial Tuning Multi-agent System Makes a Robust Retrieval-Augmented Generator (EMNLP 2024) [[paper](https://arxiv.org/abs/2405.18111)] [[code](https://github.com/chuhac/ATM-RAG)]
- SWIFTSAGE: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks (NeurIPS 2023) [[paper](https://arxiv.org/abs/2305.17390)] [[code](https://github.com/SwiftSage/SwiftSage)]
- NLRL: Natural Language Reinforcement Learning (arXiv 2024) [[paper](https://arxiv.org/abs/2411.14251)] [[code](https://github.com/waterhorse1/Natural-language-RL)]
- AGILE: A Novel Reinforcement Learning Framework of LLM Agents (NeurIPS 2024) [[paper](https://arxiv.org/abs/2405.14751)] [[code](https://github.com/bytarnish/AGILE)]
- COEVOL: Constructing Better Responses for Instruction Finetuning through Multi-Agent Cooperation (arXiv 2024) [[paper](https://arxiv.org/abs/2406.07054)] [[code](https://github.com/lirenhao1997/CoEvol)]

## Reinforcement Learning-based

- CMAT: A Multi-Agent Collaboration Tuning Framework for Enhancing Small Language Models  (arXiv 2024) [[paper](https://arxiv.org/pdf/2404.01663)] [[code](https://github.com/heimy2000/CMAT)]
- From Novice to Expert: LLM Agent Policy Optimization via Step-wise Reinforcement Learning (arXiv 2024) [[paper](https://arxiv.org/abs/2411.03817)]
- WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning (arXiv 2024) [[paper](https://arxiv.org/abs/2411.02337)] [[code](https://github.com/THUDM/WebRL)]
- SaySelf: Teaching LLMs to Express Confidence with Self-Reflective Rationales (EMNLP 2024) [[paper](https://arxiv.org/abs/2405.20974)] [[code](https://github.com/tianyang-x/SaySelf)]
- AgentGym: Evolving Large Language Model-based Agents across Diverse Environments (arXiv 2024) [[paper](https://arxiv.org/abs/2406.04151)] [[code](https://github.com/WooooDyy/AgentGym)]
- Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning (arXiv 2024) [[paper](https://arxiv.org/abs/2410.06101)]
- GELI: Global Reward to Local Rewards: Multimodal-Guided Decomposition for Improving Dialogue Agents (EMNLP 2024) [[paper](https://aclanthology.org/2024.emnlp-main.881/)]
- AGILE: A Novel Reinforcement Learning Framework of LLM Agents (NeurIPS 2024) [[paper](https://arxiv.org/abs/2405.14751)] [[code](https://github.com/bytarnish/AGILE)]
- Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents (arxiv) [[paper](https://arxiv.org/abs/2408.07199)]
- DMPO: Direct Multi-Turn Preference Optimization for Language Agents (EMNLP 2024) [[paper](https://arxiv.org/abs/2406.14868)] [[code](https://github.com/swt-user/DMPO)]
- Re-ReST: Reflection-Reinforced Self-Training for Language Agents (EMNLP 2024) [[paper](https://arxiv.org/abs/2406.01495)] [[code](https://github.com/PlusLabNLP/Re-ReST)]
- ATM: Adversarial Tuning Multi-agent System Makes a Robust Retrieval-Augmented Generator (EMNLP 2024) [[paper](https://arxiv.org/abs/2405.18111)] [[code](https://github.com/chuhac/ATM-RAG)]
- OPTIMA: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System (arXiv 2024) [[paper](https://arxiv.org/abs/2410.08115)] [[code](https://chenweize1998.github.io/optima-project-page/)]
- EPO: Hierarchical LLM Agents with Environment Preference Optimization (EMNLP 2024) [[paper](https://arxiv.org/abs/2408.16090)] [[code](https://github.com/kevinz8866/EPO)]
- Watch Every Step! LLM Agent Learning via Iterative Step-Level Process Refinement (EMNLP 2024) [[paper](https://arxiv.org/abs/2406.11176)] [[code](https://github.com/WeiminXiong/IPR)]
- AMOR: A Recipe for Building Adaptable Modular Knowledge Agents Through Process Feedback (NeurIPS 2024) [[paper](https://arxiv.org/abs/2402.01469)] [[code](https://github.com/JianGuanTHU/AMOR)]

## Hybrid Fine-Tuning Optimization

- ReFT: Reasoning with Reinforced Fine-Tuning (ACL 2024) [[paper](https://arxiv.org/abs/2401.08967)] [[code](https://github.com/lqtrung1998/mwp_ReFT)]
- AgentGym: Evolving Large Language Model-based Agents across Diverse Environments (arXiv 2024) [[paper](https://arxiv.org/abs/2406.04151)] [[code](https://github.com/WooooDyy/AgentGym)]
- AGILE: A Novel Reinforcement Learning Framework of LLM Agents (NeurIPS 2024) [[paper](https://arxiv.org/abs/2405.14751)] [[code](https://github.com/bytarnish/AGILE)]
- Re-ReST: Reflection-Reinforced Self-Training for Language Agents (EMNLP 2024) [[paper](https://arxiv.org/abs/2406.01495)] [[code](https://github.com/PlusLabNLP/Re-ReST)]
- AMOR: A Recipe for Building Adaptable Modular Knowledge Agents Through Process Feedback (NeurIPS 2024) [[paper](https://arxiv.org/abs/2402.01469)] [[code](https://github.com/JianGuanTHU/AMOR)]
- Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents (ACL 2024) [[paper](https://aclanthology.org/2024.acl-long.409.pdf)] [[code](https://github.com/Yifan-Song793/ETO)]
- OPTIMA: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System (arXiv 2024) [[paper](https://arxiv.org/abs/2410.08115)] [[code](https://chenweize1998.github.io/optima-project-page/)]
- Watch Every Step! LLM Agent Learning via Iterative Step-Level Process Refinement (EMNLP 2024) [[paper](https://arxiv.org/abs/2406.11176)] [[code](https://github.com/WeiminXiong/IPR)]
- Retrospex: Language Agent Meets Offline Reinforcement Learning Critic (EMNLP 2024) [[paper](https://aclanthology.org/2024.emnlp-main.268/)] [[code](https://github.com/Yufei-Xiang/Retrospex)]
- ENVISION:Interactive Evolution: A Neural-Symbolic Self-Training Framework For Large Language Models (arXiv 2024) [[paper](https://arxiv.org/abs/2406.11736)] [[code](https://github.com/xufangzhi/ENVISIONS)]

# Parameter-Free Optimization

## Experience-based 

- Optimus-1: Hybrid Multimodal Memory Empowered Agents Excel in Long-Horizon Tasks (NeurIPS 2024) [[paper](https://arxiv.org/abs/2408.03615)] [[code](https://cybertronagent.github.io/Optimus-1.github.io/)]
- Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents (arXiv 2024) [[paper](https://arxiv.org/abs/2405.02957)]
- ExpeL: LLM Agents Are Experiential Learners (AAAI 2024) [[paper](https://arxiv.org/abs/2308.10144)] [[code](https://github.com/LeapLabTHU/ExpeL)]
- AutoManual: Constructing Instruction Manuals by LLM Agents via Interactive Environmental Learning (NeurIPS 2024) [[paper](https://arxiv.org/abs/2405.16247)] [[code](https://github.com/minghchen/automanual)]
- AutoGuide: Automated Generation and Selection of Context-Aware Guidelines for Large Language Model Agents (NeurIPS 2024) [[paper](https://arxiv.org/abs/2403.08978)]
- Experiential Co-Learning of Software-Developing Agents (ACL 2024) [[paper](https://arxiv.org/abs/2312.17025)] [[code](https://github.com/OpenBMB/ChatDev)]

## Feedback-based 

- Reflexion: Language Agents with Verbal Reinforcement Learning (NeurIPS 2023) [[paper](https://arxiv.org/pdf/2303.11366)] [[code](https://github.com/noahshinn/reflexion)]
- QueryAgent: A Reliable and Efficient Reasoning Framework with Environmental Feedback based Self-Correction (ACL 2024) [[paper](https://arxiv.org/abs/2403.11886)] [[code](https://github.com/cdhx/QueryAgent)]
- Agent-Pro: Learning to Evolve via Policy-Level Reflection and Optimization (ACL 2024) [[paper](https://arxiv.org/abs/2402.17574)] [[code](https://github.com/zwq2018/Agent-Pro)]
- SAGE: Self-Evolving Agents with Reflective and Memory-Augmented Abilities (arXiv 2024) [[paper](https://arxiv.org/abs/2409.00872)]
- ReCon: Boosting LLM Agents with Recursive Contemplation for Effective Deception Handling (ACL-findings 2024) [[paper](https://openreview.net/pdf?id=LO-NO1-PwJR)]
- Symbolic Learning Enables Self-Evolving Agents (arXiv 2024) [[paper](https://arxiv.org/abs/2406.18532)] [[code](https://github.com/aiwaves-cn/agents)]
- COPPR:Reflective Multi-Agent Collaboration based on Large Language Models (NeurIPS 2024) [[paper](https://neurips.cc/virtual/2024/poster/93147)]
- METAREFLECTION: Learning Instructions for Language Agents using Past Reflections (EMNLP 2024) [[paper](https://arxiv.org/abs/2405.13009)] [[code](https://github.com/microsoft/prose/tree/main/misc/MetaReflection)]
- InteRecAgent: Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations (arXiv 2023) [[paper](https://arxiv.org/abs/2308.16505)] [[code](https://github.com/microsoft/RecAI/tree/main/InteRecAgent)]
- NLRL: Natural Language Reinforcement Learning (arXiv 2024) [[paper](https://arxiv.org/abs/2411.14251)] [[code](https://github.com/waterhorse1/Natural-language-RL)]
- Chain-of-Experts: When LLMs Meet Complex Operation Research Problems (ICLR 2024) [[paper](https://openreview.net/forum?id=HobyL1B9CZ)] [[code](https://github.com/xzymustbexzy/Chain-of-Experts)]
- Retroformer: Retrospective Large Language Agents with Policy Gradient Optimization (arXiv 2024) [[paper](https://arxiv.org/abs/2308.02151)]
- SELF-TUNING: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching (arXiv 2024) [[paper](https://arxiv.org/pdf/2406.06326)]
- OPRO: LARGE LANGUAGE MODELS AS OPTIMIZERS (ICLR 2024) [[paper](https://arxiv.org/abs/2309.03409)] [[code](https://github.com/google-deepmind/opro)]

## Tool-based

- Middleware for LLMs: Tools Are Instrumental for Language Agents in Complex Environments (**EMNLP 2024**) [[paper](https://aclanthology.org/2024.emnlp-main.436/)] [[code](https://github.com/OSU-NLP-Group/Middleware)]
- AVATAR: Optimizing LLM Agents for Tool-Assisted Knowledge Retrieval (**NeurIPS 2024**) [[paper](https://arxiv.org/abs/2406.11200)] [[code](https://github.com/zou-group/avatar)]
- AUTOACT: Automatic Agent Learning from Scratch for QA via Self-Planning (**ACL** **2024**) [[paper](https://arxiv.org/abs/2401.05268)] [[code](https://github.com/zjunlp/AutoAct)]
- TPTU: Large Language Model-based AI Agents for Task Planning and Tool Usage (**arXiv** **2023**) [[paper](https://arxiv.org/abs/2308.03427)]
- Lyra: Orchestrating Dual Correction in Automated Theorem Proving (**TMLR 2024**) [[paper](https://arxiv.org/abs/2309.15806)] [[code](https://github.com/chuanyang-zheng/lyra-theorem-prover)]
- Offline Training of Language Model Agents with Functions as Learnable Weights (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2402.11359)]
- VideoAgent: A Memory-Augmented Multimodal Agent for Video Understanding (**ECCV 2024**) [[paper](https://arxiv.org/pdf/2403.11481)] [[code](https://videoagent.github.io)]

## RAG-based 

- Crafting Personalized Agents through Retrieval-Augmented Generation on Editable Memory Graphs (**EMNLP 2024**) [[paper](https://arxiv.org/abs/2409.19401)]
- RaDA: Retrieval-augmented Web Agent Planning with LLMs (**ACL** **2024-findings**) [[paper](https://aclanthology.org/2024.findings-acl.802/)] [[code](https://github.com/ldilab/RaDA)]
- AutoRAG: Automated Framework for Optimization of Retrieval Augmented Generation Pipeline (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2410.20878)] [[code](https://github.com/Marker-Inc-Korea/AutoRAG_ARAGOG_Paper)]
- RAP: Retrieval-Augmented Planning with Contextual Memory for Multimodal LLM Agents (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2402.03610)] [[code](https://github.com/PanasonicConnect/rap)]
- MALADE: Orchestration of LLM-powered Agents with Retrieval Augmented Generation for Pharmacovigilance (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2408.01869v1)] [[code](https://github.com/jihyechoi77/malade)]
- PaperQA: Retrieval-Augmented Generative Agent for Scientific Research (**arXiv 2023**) [[paper](https://arxiv.org/abs/2312.07559)] [[code](https://github.com/future-house/paper-qa)]

## Multi-Agent 

- CAPO: Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation (arXiv 2024) [[paper](https://arxiv.org/abs/2411.04679)]
- A Multi-AI Agent System for Autonomous Optimization of Agentic AI Solutions via Iterative Refinement and LLM-Driven Feedback Loops (arXiv 2024) [[paper](https://arxiv.org/abs/2412.17149)] [[code](https://anonymous.4open.science/r/evolver-1D11/)]
- Training Agents with Weakly Supervised Feedback from Large Language Models (arXiv 2024) [[paper](https://arxiv.org/abs/2411.19547)]
- Chatdev: Communicative Agents for Software Development (**arXiv** **2023**) [[paper](https://arxiv.org/abs/2307.07924)] [[code](https://github.com/OpenBMB/ChatDev)]
- MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework (**arXiv** **2023**) [[paper](https://arxiv.org/abs/2308.00352)] [[code](https://github.com/geekan/MetaGPT)]
- MapCoder: Multi-Agent Code Generation for Competitive Problem Solving (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2405.11403)] [[code](https://github.com/Md-Ashraful-Pramanik/MapCoder)]
- A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration (**COLM 2024**) [[paper](https://arxiv.org/abs/2310.02170)] [[code](https://github.com/SALT-NLP/DyLAN)]
- Scaling Large-Language-Model-based Multi-Agent Collaboration (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2406.07155)] [[code](https://github.com/OpenBMB/ChatDev)]
- AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors (**arXiv** **2023**) [[paper](https://arxiv.org/abs/2308.10848)] [[code](https://github.com/OpenBMB/AgentVerse)]
- SMoA: Improving Multi-Agent Large Language Models with Sparse Mixture-of-Agents (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2411.03284)] [[code](https://github.com/David-Li0406/SMoA)]
- Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate (**EMNLP 2024**) [[paper](https://arxiv.org/abs/2305.19118)] [[code](https://github.com/Skytliang/Multi-Agents-Debate)]
- AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation (**arXiv** **2023**) [[paper](https://arxiv.org/abs/2308.08155)] [[code](https://github.com/microsoft/autogen)]

# Datasets and Benchmarks

## Datasets and Benchmarks for Evaluation

### General Evaluation Tasks

1. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents (NeurIPS 2022) [[paper](https://arxiv.org/abs/2207.01206)] [[code](https://webshop-pnlp.github.io/)]
2. WebArena: A Realistic Web Environment for Building Autonomous Agents (arXiv 2024) [[paper](https://arxiv.org/abs/2307.13854)] [[code](https://webarena.dev/)]
3. Mind2Web: Towards a Generalist Agent for the Web (NeurIPS 2023 Spotlight) [[paper](https://arxiv.org/abs/2306.06070)] [[code](https://osu-nlp-group.github.io/Mind2Web/)]
4. Reinforcement Learning on Web Interfaces using Workflow-Guided Exploration (ICLR 2018) [[paper](https://arxiv.org/abs/1802.08802)] [[code](https://github.com/Farama-Foundation/miniwob-plusplus)]
5. ScienceWorld: Is your Agent Smarter than a 5th Grader? (EMNLP 2022) [[paper](https://aclanthology.org/2022.emnlp-main.775/)] [[code](https://sciworld.apps.allenai.org/)]
6. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning (ICLR 2021) [[paper](https://arxiv.org/abs/2010.03768)] [[code](https://alfworld.github.io/)]
7. Building Cooperative Embodied Agents Modularly with Large Language Models (ICLR 2024) [[paper](https://arxiv.org/abs/2307.02485)] [[code](https://vis-www.cs.umass.edu/Co-LLM-Agents/)]
8. ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks (CVPR 2020) [[paper](https://arxiv.org/abs/1912.01734)] [[code](https://askforalfred.com/)]
9. RLCard: A Toolkit for Reinforcement Learning in Card Games (AAAI-Workshop 2020) [[paper](https://arxiv.org/abs/1910.04376)] [[code](https://github.com/datamllab/rlcard)]
10. OpenSpiel: A Framework for Reinforcement Learning in Games (arXiv 2019) [[paper](https://arxiv.org/abs/1908.09453)] [[code](https://github.com/google-deepmind/open_spiel)]
11. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering (EMNLP 2018) [[paper](https://arxiv.org/abs/1809.09600)] [[code](https://hotpotqa.github.io/)]
12. StrategyQA: Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies (TACL 2021) [[paper](https://arxiv.org/abs/2101.02235)] [[code](https://github.com/eladsegal/strategyqa)]
13. mmlu:Measuring Massive Multitask Language Understanding (ICLR 2021) [[paper](https://arxiv.org/abs/2009.03300)] [[code](https://github.com/hendrycks/test)]
14. TruthfulQA: Measuring How Models Mimic Human Falsehoods (ACL 2022) [[paper](https://arxiv.org/abs/2109.07958)] [[code](https://github.com/sylinrl/TruthfulQA)]
15. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension (ACL 2017) [[paper](https://aclanthology.org/P17-1147/)]
16. PubMedQA: A Dataset for Biomedical Research Question Answering (EMNLP 2019) [[paper](https://arxiv.org/abs/1909.06146)] [[code](https://pubmedqa.github.io/)]
17. MuSiQue: Multihop Questions via Single-hop Question Composition (TACL 2022) [[paper](https://arxiv.org/abs/2108.00573)] [[code](https://github.com/stonybrooknlp/musique)]
18. Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps (COLING 2020) [[paper](https://arxiv.org/abs/2011.01060)] [[code](https://github.com/Alab-NII/2wikimultihop)]
19. A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers (NAACL 2021) [[paper](https://arxiv.org/abs/2105.03011)] [[code](https://huggingface.co/datasets/allenai/qasper)]
20. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge (arXiv 2018) [[paper](https://arxiv.org/abs/1803.05457)] [[code](https://huggingface.co/datasets/allenai/ai2_arc)]
21. Training Verifiers to Solve Math Word Problems (arXiv 2021) [[[paper](https://arxiv.org/abs/2110.14168)] [[code](https://openai.com/index/solving-math-word-problems/)]
22. A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers (ACL 2020) [[paper](https://arxiv.org/abs/2106.15772)] [[code](https://github.com/chaochun/nlu-asdiv-dataset)]
23. mwp:Are NLP Models Really Able to Solve Simple Math Word Problems? (NAACL 2021) [[paper](https://arxiv.org/abs/2103.07191)] [[code](https://github.com/arkilpatel/SVAMP)]
24. Measuring Mathematical Problem Solving with the MATH Dataset (NeurIPS 2021) [[paper](https://arxiv.org/abs/2103.03874)] [[code](https://github.com/hendrycks/math/)]
25. T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step (ACL 2024) [[paper](https://arxiv.org/abs/2312.14033)] [[code](https://open-compass.github.io/T-Eval/)]
26. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs (ICLR 2024) [[paper](https://arxiv.org/abs/2307.16789v2)] [[code](https://github.com/OpenBMB/ToolBench)]
27. MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback (ICLR 2024) [[paper](https://arxiv.org/abs/2309.10691)] [[code](https://xwang.dev/mint-bench/)]
28. API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs (EMNLP 2023) [[paper](https://arxiv.org/abs/2304.08244)] [[code](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)]
29. A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge (ECCV 2022) [[paper](https://arxiv.org/abs/2206.01718)] [[code](https://github.com/allenai/aokvqa)]
30. Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering (NeurIPS 2022) [[paper](https://arxiv.org/abs/2209.09513)] [[code](https://scienceqa.github.io/)]
31. VQA: Visual Question Answering (ICCV 2015) [[paper](https://arxiv.org/abs/1505.00468)] [[code](https://visualqa.org/)]
32. EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding (NeurIPS 2023) [[paper](https://arxiv.org/abs/2308.09126)] [[code](https://egoschema.github.io/)]
33. NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions (CVPR 2021) [[paper](https://arxiv.org/abs/2105.08276)] [[code](https://github.com/doc-doc/NExT-QA)]
34. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (ICLR 2024) [[paper](https://arxiv.org/abs/2310.06770)] [[code](https://github.com/swe-bench/SWE-bench)]
35. Evaluating Large Language Models Trained on Code (arXiv 2021) [[paper](https://arxiv.org/abs/2107.03374)] [[code](https://github.com/openai/human-eval)]
36. LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code (arXiv 2024) [[paper](https://arxiv.org/abs/2403.07974v2)] [[code](https://github.com/LiveCodeBench/LiveCodeBench)]
37. Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs (NeurIPS 2023) [[paper](https://arxiv.org/abs/2305.03111)] [[code](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird)]
38. InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback (NeurIPS 2023) [[paper](https://arxiv.org/abs/2306.14898)] [[code](https://github.com/princeton-nlp/intercode)]

### Multi-task Benchmarks

- AgentBench: Evaluating LLMs as Agents (**ICLR** **2024**) [[paper](https://arxiv.org/abs/2308.03688)] [[code](https://github.com/THUDM/AgentBench)]
- AgentGym: Evolving Large Language Model-based Agents across Diverse Environments (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2406.04151)] [[code](https://github.com/WooooDyy/AgentGym)]
- Just-Eval: The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning (**ICLR** **2024**) [[paper](https://arxiv.org/abs/2312.01552)] [[code](https://github.com/Re-Align/just-eval)]
- StreamBench: Towards Benchmarking Continuous Improvement of Language Agents (**NeurIPS 2024**) [[paper](https://arxiv.org/abs/2406.08747)] [[code](https://github.com/stream-bench/stream-bench)]
- AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agent (**NeurIPS 2024**) [[paper](https://arxiv.org/abs/2401.13178)] [[code](https://github.com/hkust-nlp/AgentBoard)]

# Application

## Healthcare

- Med-PaLM: Large language models encode clinical knowledge (**Nature 2023**) [[paper](https://arxiv.org/abs/2212.13138)]
- DoctorGLM: Fine-tuning your Chinese Doctor is not a Herculean Task (**arXiv** **2023**) [[paper](https://arxiv.org/abs/2304.01097)] [[code](https://github.com/xionghonglin/DoctorGLM)]
- BianQue: Balancing the Questioning and Suggestion Ability of Health LLMs with Multi-turn Health Conversations Polished by ChatGPT (**arXiv** **2023**) [[paper](https://arxiv.org/abs/2310.15896)] [[code](https://github.com/scutcyr/BianQue)]
- DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation (**arXiv** **2023**) [[paper](https://arxiv.org/abs/2308.14346)] [[code](https://github.com/FudanDISC/DISC-MedLLM)]
- ClinicalAgent: Clinical Trial Multi-Agent System with Large Language Model-based Reasoning (**BCB 2024**) [[paper](https://dl.acm.org/doi/abs/10.1145/3698587.3701359)] [[code](https://github.com/LeoYML/clinical-agent)]
- MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning (**ACL-findings 2024**) [[paper](https://aclanthology.org/2024.findings-acl.33/)] [[code](https://github.com/gersteinlab/MedAgents)]
- MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making (**NeurIPS 2024**) [[paper](https://arxiv.org/abs/2404.15155)] [[code](https://github.com/mitmedialab/MDAgents)]
- Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2405.02957)]
- AI Hospital: Benchmarking Large Language Models in a Multi-agent Medical Interaction Simulator (**COLING 2025**) [[paper](https://arxiv.org/abs/2402.09742)] [[code](https://github.com/LibertFan/AI_Hospital)]
- KG4Diagnosis: A Hierarchical Multi-Agent LLM Framework with Knowledge Graph Enhancement for Medical Diagnosis (**AAAI-25 Bridge Program**) [[paper](https://arxiv.org/abs/2412.16833)]
- AgentMD: Empowering Language Agents for Risk Prediction with Large-Scale Clinical Tool Learning (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2402.13225)] [[code](https://github.com/ncbi-nlp/Clinical-Tool-Learning)]
- MMedAgent: Learning to Use Medical Tools with Multi-modal Agent (**EMNLP-findings 2024**) [[paper](https://arxiv.org/abs/2407.02483)] [[code](https://github.com/Wangyixinxin/MMedAgent)]
- HuatuoGPT-o1: Towards Medical Complex Reasoning with LLMs (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2412.18925)] [[code](https://github.com/FreedomIntelligence/HuatuoGPT-o1)]
- IIMedGPT: Promoting Large Language Model Capabilities of Medical Tasks by Efficient Human Preference Alignment (**arXiv** **2025**) [[paper](https://arxiv.org/abs/2501.02869)]

## Science

- CellAgent: An LLM-driven Multi-Agent Framework for Automated Single-cell Data Analysis (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2407.09811)] [[code](https://github.com/lsq2wal/CellAgent)]
- BioDiscoveryAgent: An AI Agent for Designing Genetic Perturbation Experiments (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2405.17631)] [[code](https://github.com/snap-stanford/BioDiscoveryAgent/)]
- ProtAgents: Protein discovery via large language model multi-agent collaborations combining physics and machine learning (**Digital Discovery 2024**) [[paper](https://arxiv.org/abs/2402.04268)] [[code](https://github.com/lamm-mit/ProtAgents)]
- CRISPR-GPT: An LLM Agent for Automated Design of Gene-Editing Experiments (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2404.18021)] [[code](https://github.com/cong-lab/crispr-gpt-pub)]
- ChemCrow: Augmenting Large-Language Models with Chemistry Tools (**Nature** **Machine Intelligence** **2024**) [[paper](https://arxiv.org/abs/2304.05376)] [[code](https://github.com/ur-whitelab/chemcrow-public)]
- DrugAssist: A Large Language Model for Molecule Optimization (**Briefings in Bioinformatics 2024**) [[paper](https://arxiv.org/abs/2401.10334)] [[code](https://github.com/blazerye/DrugAssist)]
- Agent-based Learning of Materials Datasets from Scientific Literature (**Digital Discovery 2024**) [[paper](https://arxiv.org/abs/2312.11690)] [[code](https://github.com/AI4ChemS/Eunomia)]
- DrugAgent: Automating AI-aided Drug Discovery Programming through LLM Multi-Agent Collaboration (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2411.15692)] [[code](https://github.com/anrohanro/DrugAgent)]
- MProt-DPO: Breaking the ExaFLOPS Barrier for Multimodal Protein Design Workflows with Direct Preference Optimization (**SC 2024**) [[paper](https://dl.acm.org/doi/10.1109/SC41406.2024.00013)]
- Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2410.09403)] [[code](https://github.com/open-sciencelab/Virtual-Scientists)]
- SciAgent: Tool-augmented Language Models for Scientific Reasoning (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2402.11451)]

## Embodied Intelligence

- Building Cooperative Embodied Agents Modularly with Large Language Models (**ICLR** **2024**) [[paper](https://arxiv.org/abs/2307.02485)] [[code](https://vis-www.cs.umass.edu/Co-LLM-Agents/)]
- Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (**PMLR 2023**) [[paper](https://arxiv.org/abs/2204.01691)] [[code](https://github.com/google-research/google-research/tree/master/saycan)]
- RoCo: Dialectic Multi-Robot Collaboration with Large Language Models (**ICRA 2024**) [[paper](https://arxiv.org/abs/2307.04738)] [[code](https://github.com/MandiZhao/robot-collab)]
- Voyager: An Open-Ended Embodied Agent with Large Language Models (**arXiv** **2023**) [[paper](https://arxiv.org/abs/2305.16291)] [[code](https://github.com/MineDojo/Voyager)]
- MultiPLY: A Multisensory Object-Centric Embodied Large Language Model in 3D World (**CVPR** **2024**) [[paper](https://arxiv.org/abs/2401.08577)] [[code](https://github.com/UMass-Foundation-Model/MultiPLY)]
- Retrospex: Language Agent Meets Offline Reinforcement Learning Critic (**EMNLP 2024**) [[paper](https://aclanthology.org/2024.emnlp-main.268/)] [[code](https://github.com/Yufei-Xiang/Retrospex)]
- EPO: Hierarchical LLM Agents with Environment Preference Optimization (**EMNLP 2024**) [[paper](https://arxiv.org/abs/2408.16090)] [[code](https://github.com/kevinz8866/EPO)]
- AutoManual: Constructing Instruction Manuals by LLM Agents via Interactive Environmental Learning (**NeurIPS 2024**) [[paper](https://arxiv.org/abs/2405.16247)] [[code](https://github.com/minghchen/automanual)]
- MSI-Agent: Incorporating Multi-Scale Insight into Embodied Agents for Superior Planning and Decision-Making (**EMNLP 2024**) [[paper](https://arxiv.org/abs/2409.16686)]
- iVideoGPT: Interactive VideoGPTs are Scalable World Models (**NeurIPS 2024**) [[paper](https://arxiv.org/abs/2405.15223)] [[code](https://github.com/thuml/iVideoGPT)]
- AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2401.12963)]
- Embodied Multi-Modal Agent trained by an LLM from a Parallel TextWorld (**CVPR** **2024**) [[paper](https://arxiv.org/abs/2311.16714)] [[code](https://github.com/stevenyangyj/Emma-Alfworld)]

## Finance

- Large Language Model Agent in Financial Trading: A Survey (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2408.06361)]
- TradingGPT: Multi-Agent System with Layered Memory and Distinct Characters for Enhanced Financial Trading Performance (**arXiv** **2023**) [[paper](https://arxiv.org/abs/2309.03736)]
- FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design (**AAAI-SS**) [[paper](https://arxiv.org/abs/2311.13743)] [[code](https://github.com/pipiku915/FinMem-LLM-StockTrading)]
- A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist (**SIGKDD 2024**) [[paper](https://arxiv.org/abs/2402.18485)]
- Learning to Generate Explainable Stock Predictions using Self-Reflective Large Language Models (**WWW 2024**) [[paper](https://arxiv.org/abs/2402.03659)] [[code](https://github.com/koa-fin/sep)]
- FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making (**NeurIPS 2024**) [[paper](https://arxiv.org/abs/2407.06567)]
- TradingAgents: Multi-Agents LLM Financial Trading Framework (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2412.20138)] [[code](https://github.com/TradingAgents-AI/TradingAgents)]
- FinVision: A Multi-Agent Framework for Stock Market Prediction (**ICAIF 2024**) [[paper](https://arxiv.org/abs/2411.08899)]
- Simulating Financial Market via Large Language Model-based Agents (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2406.19966)]
- FinVerse: An Autonomous Agent System for Versatile Financial Analysis (**arXiv 2024**) [[paper](https://arxiv.org/abs/2406.06379)]
- FinRobot: An Open-Source AI Agent Platform for Financial Applications using Large Language Models (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2405.14767)] [[code](https://github.com/AI4Finance-Foundation/FinRobot)]

## Programming

- Agents in Software Engineering: Survey, Landscape, and Vision (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2409.09030)] [[code](https://github.com/DeepSoftwareAnalytics/Awesome-Agent4SE)]
- Large Language Model-Based Agents for Software Engineering: A Survey (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2409.02977)] [[code](https://github.com/FudanSELab/Agent4SE-Paper-List)]
- Chatdev: Communicative Agents for Software Development (**ACL 2024**) [[paper](https://arxiv.org/abs/2307.07924)] [[code](https://github.com/OpenBMB/ChatDev)]
- MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework (**ICLR** **2024**) [[paper](https://arxiv.org/abs/2308.00352)] [[code](https://github.com/geekan/MetaGPT)]
- MapCoder: Multi-Agent Code Generation for Competitive Problem Solving (**ACL** **2024**) [[paper](https://arxiv.org/abs/2405.11403)] [[code](https://github.com/Md-Ashraful-Pramanik/MapCoder)]
- Self-Organized Agents: A LLM Multi-Agent Framework toward Ultra Large-Scale Code Generation and Optimization (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2404.02183)] [[code](https://github.com/tsukushiAI/self-organized-agent)]
- Multi-Agent Software Development through Cross-Team Collaboration (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2406.08979)] [[code](https://github.com/OpenBMB/ChatDev)]
- SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering (**NeurIPS 2024**) [[paper](http://arxiv.org/abs/2405.15793)] [[code](https://swe-agent.com/latest/)]
- CodeAgent: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level Coding Challenges (**ACL** **2024**) [[paper](https://arxiv.org/abs/2401.07339)]
- AgentCoder: Multi-Agent-based Code Generation with Iterative Testing and Optimisation (**arXiv** **2023**) [[paper](https://arxiv.org/abs/2312.13010)] [[code](https://github.com/huangd1999/AgentCoder)]
- RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning (**arXiv** **2024**) [[paper](https://arxiv.org/abs/2410.02089)]
- Lemur: Harmonizing Natural Language and Code for Language Agents (**ICLR** **2024**) [[paper](https://arxiv.org/abs/2310.06830)] [[code](https://github.com/OpenLemur/Lemur)]
- AgileCoder: Dynamic Collaborative Agents for Software Development based on Agile Methodology (**FORGE 2025**) [[paper](https://arxiv.org/abs/2406.11912)] [[code](https://github.com/FSoft-AI4Code/AgileCoder)]


 ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=YoungDubbyDu/LLM-Agent-Optimization&type=Date)](https://www.star-history.com/#YoungDubbyDu/LLM-Agent-Optimization&Date)
