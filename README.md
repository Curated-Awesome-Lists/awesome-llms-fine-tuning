# Awesome LLM Fine-Tuning

Welcome to the curated collection of resources for fine-tuning Large Language Models (LLMs) like GPT, BERT, RoBERTa, and their numerous variants! In this era of artificial intelligence, the ability to adapt pre-trained models to specific tasks and domains has become an indispensable skill for researchers, data scientists, and machine learning practitioners.

Large Language Models, trained on massive datasets, capture an extensive range of knowledge and linguistic nuances. However, to unleash their full potential in specific applications, fine-tuning them on targeted datasets is paramount. This process not only enhances the models’ performance but also ensures that they align with the particular context, terminology, and requirements of the task at hand.

In this awesome list, we have meticulously compiled a range of resources, including tutorials, papers, tools, frameworks, and best practices, to aid you in your fine-tuning journey. Whether you are a seasoned practitioner looking to expand your expertise or a beginner eager to step into the world of LLMs, this repository is designed to provide valuable insights and guidelines to streamline your endeavors.

## Table of Contents

- [GitHub projects](#github-projects)
- [Articles & Blogs](#articles-&-blogs)
- [Online Courses](#online-courses)
- [Books](#books)
- [Research Papers](#research-papers)
- [Videos](#videos)
- [Tools & Software](#tools-&-software)
- [Conferences & Events](#conferences-&-events)
- [Slides & Presentations](#slides-&-presentations)
- [Podcasts](#podcasts)

## GitHub projects

- [LlamaIndex](https://github.com/run-llama/llama_index) 🦙: A data framework for your LLM applications. (23010 stars)
- [Petals](https://github.com/bigscience-workshop/petals) 🌸: Run LLMs at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading. (7768 stars)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): An easy-to-use LLM fine-tuning framework (LLaMA-2, BLOOM, Falcon, Baichuan, Qwen, ChatGLM3). (5532 stars)
- [lit-gpt](https://github.com/Lightning-AI/lit-gpt): Hackable implementation of state-of-the-art open-source LLMs based on nanoGPT. Supports flash attention, 4-bit and 8-bit quantization, LoRA and LLaMA-Adapter fine-tuning, pre-training. Apache 2.0-licensed. (3469 stars)
- [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio): A framework and no-code GUI for fine-tuning LLMs. Documentation: [https://h2oai.github.io/h2o-llmstudio/](https://h2oai.github.io/h2o-llmstudio/) (2880 stars)
- [Phoenix](https://github.com/Arize-ai/phoenix): AI Observability & Evaluation - Evaluate, troubleshoot, and fine tune your LLM, CV, and NLP models in a notebook. (1596 stars)
- [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters): Code for the EMNLP 2023 Paper: "LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models". (769 stars)
- [Platypus](https://github.com/arielnlee/Platypus): Code for fine-tuning Platypus fam LLMs using LoRA. (589 stars)
- [xtuner](https://github.com/InternLM/xtuner): A toolkit for efficiently fine-tuning LLM (InternLM, Llama, Baichuan, QWen, ChatGLM2). (540 stars)
- [DB-GPT-Hub](https://github.com/eosphoros-ai/DB-GPT-Hub): A repository that contains models, datasets, and fine-tuning techniques for DB-GPT, with the purpose of enhancing model performance, especially in Text-to-SQL, and achieved higher exec acc than GPT-4 in spider eval with 13B LLM used this project. (422 stars)
- [LLM-Finetuning-Hub](https://github.com/georgian-io/LLM-Finetuning-Hub) : Repository that contains LLM fine-tuning and deployment scripts along with our research findings. :star: 416
- [Finetune_LLMs](https://github.com/mallorbc/Finetune_LLMs) : Repo for fine-tuning Casual LLMs. :star: 391
- [MFTCoder](https://github.com/codefuse-ai/MFTCoder) : High Accuracy and efficiency multi-task fine-tuning framework for Code LLMs; 业内首个高精度、高效率、多任务、多模型支持、多训练算法，大模型代码能力微调框架. :star: 337
- [llmware](https://github.com/llmware-ai/llmware) : Providing enterprise-grade LLM-based development framework, tools, and fine-tuned models. :star: 289
- [LLM-Kit](https://github.com/wpydcr/LLM-Kit) : 🚀WebUI integrated platform for latest LLMs | 各大语言模型的全流程工具 WebUI 整合包。支持主流大模型API接口和开源模型。支持知识库，数据库，角色扮演，mj文生图，LoRA和全参数微调，数据集制作，live2d等全流程应用工具. :star: 232
- [h2o-wizardlm](https://github.com/h2oai/h2o-wizardlm) : Open-Source Implementation of WizardLM to turn documents into Q:A pairs for LLM fine-tuning. :star: 228
- [hcgf](https://github.com/hscspring/hcgf) : Humanable Chat Generative-model Fine-tuning | LLM微调. :star: 196
- [llm_qlora](https://github.com/georgesung/llm_qlora) : Fine-tuning LLMs using QLoRA. :star: 136
- [awesome-llm-human-preference-datasets](https://github.com/glgh/awesome-llm-human-preference-datasets) : A curated list of Human Preference Datasets for LLM fine-tuning, RLHF, and eval. :star: 124
- [llm_finetuning](https://github.com/taprosoft/llm_finetuning) : Convenient wrapper for fine-tuning and inference of Large Language Models (LLMs) with several quantization techniques (GTPQ, bitsandbytes). :star: 114

## Articles & Blogs

- [Complete Guide to LLM Fine Tuning for Beginners](https://medium.com/@mayaakim/complete-guide-to-llm-fine-tuning-for-beginners-c2c38a3252be) 📚: A comprehensive guide that explains the process of fine-tuning a pre-trained model for new tasks, covering key concepts and providing a concrete example.
- [Fine-Tuning Large Language Models (LLMs)](https://towardsdatascience.com/fine-tuning-large-language-models-llms-23473d763b91) 📖: This blog post presents an overview of fine-tuning pre-trained LLMs, discussing important concepts and providing a practical example with Python code.
- [Creating a Domain Expert LLM: A Guide to Fine-Tuning](https://hackernoon.com/creating-a-domain-expert-llm-a-guide-to-fine-tuning) 📝: An article that dives into the concept of fine-tuning using OpenAI's API, showcasing an example of fine-tuning a large language model for understanding the plot of a Handel opera.
- [A Beginner's Guide to LLM Fine-Tuning](https://towardsdatascience.com/a-beginners-guide-to-llm-fine-tuning-4bae7d4da672) 🌱: A guide that covers the process of fine-tuning LLMs, including the use of tools like QLoRA for configuring and fine-tuning models.
- [Knowledge Graphs & LLMs: Fine-Tuning Vs. Retrieval-Augmented Generation](https://medium.com/neo4j/knowledge-graphs-llms-fine-tuning-vs-retrieval-augmented-generation-30e875d63a35) 📖: This blog post explores the limitations of LLMs and provides insights into fine-tuning them in conjunction with knowledge graphs.
- [Fine-tune an LLM on your personal data: create a “The Lord of the Rings” storyteller](https://medium.com/@jeremyarancio/fine-tune-an-llm-on-your-personal-data-create-a-the-lord-of-the-rings-storyteller-6826dd614fa9) ✏️: An article that demonstrates how to train your own LLM on personal data, offering control over personal information without relying on OpenAI's GPT-4.
- [Fine-tuning an LLM model with H2O LLM Studio to generate Cypher statements](https://towardsdatascience.com/fine-tuning-an-llm-model-with-h2o-llm-studio-to-generate-cypher-statements-3f34822ad5) 🧪: This blog post provides an example of fine-tuning an LLM model using H2O LLM Studio for generating Cypher statements, enabling chatbot applications with knowledge graphs.
- [Fine-Tune Your Own Llama 2 Model in a Colab Notebook](https://towardsdatascience.com/fine-tune-your-own-llama-2-model-in-a-colab-notebook-df9823a04a32) 📝: A practical introduction to LLM fine-tuning, demonstrating how to implement it in a Google Colab notebook to create your own Llama 2 model.
- [Thinking about fine-tuning a LLM? Here's 3 considerations before you get started](https://towardsdatascience.com/thinking-about-fine-tuning-an-llm-heres-3-considerations-before-you-get-started-c1f483f293) 💡: This article discusses three ideas to consider when fine-tuning LLMs, including ways to improve GPT beyond PEFT and LoRA, and the importance of investing resources wisely.
- [Introduction to LLMs and the generative AI : Part 3—Fine Tuning LLM with instruction](https://medium.com/@yash9439/introduction-to-llms-and-the-generative-ai-part-3-fine-tuning-llm-with-instruction-and-326bc95e07ae) 📚: This article explores the role of LLMs in artificial intelligence applications and provides an overview of fine-tuning them.
- [RAG vs Finetuning — Which Is the Best Tool to Boost Your LLM Application](https://towardsdatascience.com/rag-vs-finetuning-which-is-the-best-tool-to-boost-your-llm-application-94654b1eaba7) - A blog post discussing the aspects to consider when building LLM applications and choosing the right method for your use case. 👨‍💻
- [Finetuning an LLM: RLHF and alternatives (Part I)](https://medium.com/mantisnlp/finetuning-an-llm-rlhf-and-alternatives-part-i-2106b95c8087) - An article showcasing alternative methods to RLHF, specifically Direct Preference Optimization (DPO). 🔄
- [When Should You Fine-Tune LLMs?](https://towardsdatascience.com/when-should-you-fine-tune-llms-2dddc09a404a) - Exploring the comparison between fine-tuning open-source LLMs and using a closed API for LLM Queries at Scale. 🤔
- [Fine-Tuning Large Language Models](https://cobusgreyling.medium.com/fine-tuning-large-language-models-f937869cef17) - Considering the fine-tuning of large language models and comparing it to zero and few shot approaches. 🎯
- [Private GPT: Fine-Tune LLM on Enterprise Data](https://towardsdatascience.com/private-gpt-fine-tune-llm-on-enterprise-data-7e663d808e6a) - Exploring training techniques that allow fine-tuning LLMs on smaller GPUs. 🖥️
- [Fine-tune Google PaLM 2 with Scikit-LLM](https://medium.com/@iryna230520/fine-tune-google-palm-2-with-scikit-llm-d41b0aa673a5) - Demonstrating how to fine-tune Google PaLM 2, the most advanced LLM from Google, using Scikit-LLM. 📈
- [A Deep-Dive into Fine-Tuning of Large Language Models](https://rpradeepmenon.medium.com/a-deep-dive-into-fine-tuning-of-large-language-models-96f7029ac0e1) - A comprehensive blog on fine-tuning LLMs like GPT-4 & BERT, providing insights, trends, and benefits. 🚀
- [Pre-training, fine-tuning and in-context learning in Large Language Models](https://medium.com/@atmabodha/pre-training-fine-tuning-and-in-context-learning-in-large-language-models-llms-dd483707b122) - Discussing the concepts of pre-training, fine-tuning, and in-context learning in LLMs. 📚
- [List of Open Sourced Fine-Tuned Large Language Models](https://sungkim11.medium.com/list-of-open-sourced-fine-tuned-large-language-models-llm-8d95a2e0dc76) - A curated list of open-sourced fine-tuned LLMs that can be run locally on your computer. 📋
- [Practitioners guide to fine-tune LLMs for domain-specific use case](https://cismography.medium.com/practitioners-guide-to-fine-tune-llms-for-domain-specific-use-case-part-1-4561714d874f) - A guide covering key learnings and conclusions on fine-tuning LLMs for domain-specific use cases. 📝

## Online Courses

- [Fine-Tuning Fundamentals: Unlocking the Potential of LLMs | Udemy](https://www.udemy.com/course/building-chatgpt-style-agents/): A practical course for beginners on building chatGPT-style models and adapting them for specific use cases.
- [Generative AI with Large Language Models | Coursera](https://www.coursera.org/learn/generative-ai-with-llms): Learn the fundamentals of generative AI with LLMs and how to deploy them in practical applications. Enroll for free.
- [Large Language Models: Application through Production | edX](https://www.edx.org/learn/computer-science/databricks-large-language-models-application-through-production): An advanced course for developers, data scientists, and engineers to build LLM-centric applications using popular frameworks and achieve end-to-end production readiness.
- [Finetuning Large Language Models | Coursera Guided Project](https://www.coursera.org/projects/finetuning-large-language-models-project): A short guided project that covers essential finetuning concepts and training of large language models.
- [OpenAI & ChatGPT API's: Expert Fine-tuning for Developers | Udemy](https://www.udemy.com/course/mastering-chatgpt-models-from-fine-tuning-to-deployment-openai/): Discover the power of GPT-3 in creating conversational AI solutions, including topics like prompt engineering, fine-tuning, integration, and deploying ChatGPT models.
- [Large Language Models Professional Certificate | edX](https://www.edx.org/certificates/professional-certificate/databricks-large-language-models): Learn how to build and productionize Large Language Model (LLM) based applications using the latest frameworks, techniques, and theory behind foundation models.
- [Improving the Performance of Your LLM Beyond Fine Tuning | Udemy](https://www.udemy.com/course/improving-the-performance-of-your-llm-beyond-fine-tuning/): A course designed for business leaders and developers interested in fine-tuning LLM models and exploring techniques for improving their performance.
- [Introduction to Large Language Models | Coursera](https://www.coursera.org/learn/introduction-to-large-language-models): An introductory level micro-learning course offered by Google Cloud, explaining the basics of Large Language Models (LLMs) and their use cases. Enroll for free.
- [Syllabus | LLM101x | edX](https://courses.edx.org/courses/course-v1:Databricks+LLM101x+2T2023/c861b0726ce24e099ad80111145f4217/): Learn how to use data embeddings, vector databases, and fine-tune LLMs with domain-specific data to augment LLM pipelines.
- [Performance Tuning Deep Learning Models Master Class | Udemy](https://www.udemy.com/course/performance-tuning-deep-learning-models-master-class/): A master class on tuning deep learning models, covering techniques to accelerate learning and optimize performance.
- [Best Large Language Models (LLMs) Courses & Certifications](https://www.coursera.org/courses?query=large%20language%20models): Curated from top educational institutions and industry leaders, this selection of LLMs courses aims to provide quality training for individuals and corporate teams looking to learn or improve their skills in fine-tuning LLMs.
- [Mastering Language Models: Unleashing the Power of LLMs](https://www.udemy.com/course/mastering-language-models-unleashing-the-power-of-llms/): In this comprehensive course, you'll delve into the fundamental principles of NLP and explore how LLMs have reshaped the landscape of AI applications. A comprehensive guide to advanced NLP and LLMs.
- [LLMs Mastery: Complete Guide to Transformers & Generative AI](https://www.udemy.com/course/llms-mastery-complete-guide-to-transformers-generative-ai/): This course provides a great overview of AI history and covers fine-tuning the three major LLM models: BERT, GPT, and T5. Suitable for those interested in generative AI, LLMs, and production-level applications.
- [Exploring The Technologies Behind ChatGPT, GPT4 & LLMs](https://www.udemy.com/course/exploring-the-technologies-behind-chatgpt-openai/): The only course you need to learn about large language models like ChatGPT, GPT4, BERT, and more. Gain insights into the technologies behind these LLMs.
- [Non-technical Introduction to Large Language Models](https://www.udemy.com/course/non-technical-introduction-to-large-language-models/): An overview of large language models for non-technical individuals, explaining the existing challenges and providing simple explanations without complex jargon.
- [Large Language Models: Foundation Models from the Ground Up](https://www.edx.org/learn/computer-science/databricks-large-language-models-foundation-models-from-the-ground-up): Delve into the details of foundation models in LLMs, such as BERT, GPT, and T5. Gain an understanding of the latest advances that enhance LLM functionality.

## Books

- [Generative AI with Large Language Models — New Hands-on Course by Deeplearning.ai and AWS](https://aws.amazon.com/blogs/aws/generative-ai-with-large-language-models-new-hands-on-course-by-deeplearning-ai-and-aws/)
- A hands-on course that teaches how to fine-tune Large Language Models (LLMs) using reward models and reinforcement learning, with a focus on generative AI.
- [From Data Selection To Fine Tuning: The Technical Guide To Constructing LLM Models](https://www.amazon.com/Data-Selection-Fine-Tuning-Constructing/dp/B0CH2FLV2Q)
- A technical guide that covers the process of constructing LLM models, from data selection to fine-tuning.
- [The LLM Knowledge Cookbook: From, RAG, to QLoRA, to Fine Tuning, and all the Recipes In Between!](https://www.amazon.com/LLM-Knowledge-Cookbook-Recipes-Between-ebook/dp/B0CKH9B58N)
- A comprehensive cookbook that explores various LLM models, including techniques like Retrieve and Generate (RAG) and Query Language Representation (QLoRA), as well as the fine-tuning process.
- [Principles for Fine-tuning LLMs](https://www.packtpub.com/article-hub/principles-for-fine-tuning-llms)
- An article that demystifies the process of fine-tuning LLMs and explores different techniques, such as in-context learning, classic fine-tuning methods, parameter-efficient fine-tuning, and Reinforcement Learning with Human Feedback (RLHF).
- [From Data Selection To Fine Tuning: The Technical Guide To Constructing LLM Models](https://www.goodreads.com/book/show/198555505-from-data-selection-to-fine-tuning)
- A technical guide that provides insights into building and training large language models (LLMs).
- [Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/)
- A book that covers the advancements in language AI systems driven by deep learning, focusing on large language models.
- [Fine-tune Llama 2 for text generation on Amazon SageMaker JumpStart](https://aws.amazon.com/blogs/machine-learning/fine-tune-llama-2-for-text-generation-on-amazon-sagemaker-jumpstart/)
- Learn how to fine-tune Llama 2 models using Amazon SageMaker JumpStart for optimized dialogue generation.
- [Fast and cost-effective LLaMA 2 fine-tuning with AWS Trainium](https://aws.amazon.com/blogs/machine-learning/fast-and-cost-effective-llama-2-fine-tuning-with-aws-trainium/)
- A blog post that explains how to achieve fast and cost-effective fine-tuning of LLaMA 2 models using AWS Trainium.
- [Fine-tuning - Advanced Deep Learning with Python [Book]](https://www.oreilly.com/library/view/advanced-deep-learning/9781789956177/bd91eb1d-87cf-463d-bbd3-4177a42f3da7.xhtml) 💡: A book that explores the fine-tuning task following the pretraining task in advanced deep learning with Python.
- [The LLM Knowledge Cookbook: From, RAG, to QLoRA, to Fine ...](https://www.barnesandnoble.com/w/the-llm-knowledge-cookbook-richard-anthony-aragon/1144180729) 💡: A comprehensive guide to using large language models (LLMs) for various tasks, covering everything from the basics to advanced fine-tuning techniques.
- [Quick Start Guide to Large Language Models: Strategies and Best ...](https://www.oreilly.com/library/view/quick-start-guide/9780138199425/) 💡: A guide focusing on strategies and best practices for large language models (LLMs) like BERT, T5, and ChatGPT, showcasing their unprecedented performance in various NLP tasks.
- [4. Advanced GPT-4 and ChatGPT Techniques - Developing Apps ...](https://www.oreilly.com/library/view/developing-apps-with/9781098152475/ch04.html) 💡: A chapter that dives into advanced techniques for GPT-4 and ChatGPT, including prompt engineering, zero-shot learning, few-shot learning, and task-specific fine-tuning.
- [What are Large Language Models? - LLM AI Explained - AWS](https://aws.amazon.com/what-is/large-language-model/) 💡: An explanation of large language models (LLMs), discussing the concepts of few-shot learning and fine-tuning to improve model performance.

## Research Papers

- [LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2304.01933) 📄: This paper presents LLM-Adapters, an easy-to-use framework that integrates various adapters into LLMs for parameter-efficient fine-tuning (PEFT) on different tasks.
- [Two-stage LLM Fine-tuning with Less Specialization](https://arxiv.org/abs/2211.00635) 📄: ProMoT, a two-stage fine-tuning framework, addresses the issue of format specialization in LLMs through Prompt Tuning with MOdel Tuning, improving their general in-context learning performance.
- [Fine-tuning Large Enterprise Language Models via Ontological Reasoning](https://arxiv.org/abs/2306.10723) 📄: This paper proposes a neurosymbolic architecture that combines Large Language Models (LLMs) with Enterprise Knowledge Graphs (EKGs) to achieve domain-specific fine-tuning of LLMs.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) 📄: QLoRA is an efficient finetuning approach that reduces memory usage while preserving task performance, offering insights on quantized pretrained language models.
- [Full Parameter Fine-tuning for Large Language Models with Limited Resources](https://arxiv.org/abs/2306.09782) 📄: This work introduces LOMO, a low-memory optimization technique, enabling the full parameter fine-tuning of large LLMs with limited GPU resources.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) 📄: LoRA proposes a methodology to adapt large pre-trained models to specific tasks by injecting trainable rank decomposition matrices into each layer, reducing the number of trainable parameters while maintaining model quality.
- [Enhancing LLM with Evolutionary Fine Tuning for News Summary Generation](https://arxiv.org/abs/2307.02839) 📄: This paper presents a new paradigm for news summary generation using LLMs, incorporating genetic algorithms and powerful natural language understanding capabilities.
- [How do languages influence each other? Studying cross-lingual data sharing during LLM fine-tuning](https://arxiv.org/abs/2305.13286) 📄: This study investigates cross-lingual data sharing during fine-tuning of multilingual large language models (MLLMs) and analyzes the influence of different languages on model performance.
- [Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333) 📄: MeZO, a memory-efficient zeroth-order optimizer, enables fine-tuning of large language models while significantly reducing the memory requirements.
- [Learning to Reason over Scene Graphs: A Case Study of Finetuning LLMs](https://arxiv.org/abs/2305.07716) 📄: This work explores the applicability of GPT-2 LLMs in robotic task planning, demonstrating the potential for using LLMs in long-horizon task planning scenarios.
- [Privately Fine-Tuning Large Language Models with](https://arxiv.org/abs/2210.15042): This paper explores the application of differential privacy to add privacy guarantees to fine-tuning large language models (LLMs).
- [DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Systems](http://arxiv.org/abs/2309.11325): This paper presents DISC-LawLLM, an intelligent legal system that utilizes fine-tuned LLMs with legal reasoning capability to provide a wide range of legal services.
- [Multi-Task Instruction Tuning of LLaMa for Specific Scenarios: A](https://arxiv.org/abs/2305.13225): The paper investigates the effectiveness of fine-tuning LLaMa, a foundational LLM, on specific writing tasks, demonstrating significant improvement in writing abilities.
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155): This paper proposes a method to align language models with user intent by fine-tuning them using human feedback, resulting in models preferred over larger models in human evaluations.
- [Large Language Models Can Self-Improve](https://arxiv.org/abs/2210.11610): The paper demonstrates that LLMs can self-improve their reasoning abilities by fine-tuning using self-generated solutions, achieving state-of-the-art performance without ground truth labels.
- [Embracing Large Language Models for Medical Applications](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10292051/): This paper highlights the potential of fine-tuned LLMs in medical applications, improving diagnostic accuracy and supporting clinical decision-making.
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416): The paper explores instruction fine-tuning on LLMs, demonstrating significant improvements in performance and generalization to unseen tasks.
- [Federated Fine-tuning of Billion-Sized Language Models across](https://arxiv.org/abs/2308.13894): This work introduces FwdLLM, a federated learning protocol designed to enhance the efficiency of fine-tuning large LLMs on mobile devices, improving memory and time efficiency.
- [A Comprehensive Overview of Large Language Models](https://arxiv.org/pdf/2307.06435): This paper provides an overview of the development and applications of large language models and their transfer learning capabilities.
- [Fine-tuning language models to find agreement among humans with](https://arxiv.org/abs/2211.15006): The paper explores the fine-tuning of a large LLM to generate consensus statements that maximize approval for a group of people with diverse opinions.

## Videos

- [Fine-tuning Llama 2 on Your Own Dataset | Train an LLM for Your ...](https://www.youtube.com/watch?v=MDA3LUKNl1E): Learn how to fine-tune Llama 2 model on a custom dataset.
- [Fine-tuning LLM with QLoRA on Single GPU: Training Falcon-7b on ...](https://www.youtube.com/watch?v=DcBC4yGHV4Q): This video demonstrates the fine-tuning process of the Falcon 7b LLM using QLoRA.
- [Fine-tuning an LLM using PEFT | Introduction to Large Language ...](https://www.youtube.com/watch?v=6SpZkwWuldU): Discover how to fine-tune an LLM using PEFT, a technique that requires fewer resources.
- [LLAMA-2 Open-Source LLM: Custom Fine-tuning Made Easy on a ...](https://www.youtube.com/watch?v=8cc4bJtycOA): A step-by-step guide on how to fine-tune the LLama 2 LLM model on your custom dataset.
- [New Course: Finetuning Large Language Models - YouTube](https://www.youtube.com/watch?v=9PxhCekQYNI): This video introduces a course on fine-tuning LLMs, covering model selection, data preparation, training, and evaluation.
- [Q: How to create an Instruction Dataset for Fine-tuning my LLM ...](https://www.youtube.com/watch?v=BJQrQT2Xfyo): In this tutorial, beginners learn about fine-tuning LLMs, including when, how, and why to do it.
- [LLM Module 4: Fine-tuning and Evaluating LLMs | 4.13.1 Notebook ...](https://www.youtube.com/watch?v=2vEpMb4ofVU): A notebook demo on fine-tuning and evaluating LLMs.
- [Google LLM Fine-Tuning/Adapting/Customizing - Getting Started ...](https://www.youtube.com/watch?v=cJ96rqW8L84): Get started with fine-tuning Google's PaLM 2 large language model through a step-by-step guide.
- [Pretraining vs Fine-tuning vs In-context Learning of LLM (GPT-x ...](https://www.youtube.com/watch?v=_FYwnO_g-4E): An ultimate guide explaining pretraining, fine-tuning, and in-context learning of LLMs like GPT-x.
- [How to Fine-Tune an LLM with a PDF - Langchain Tutorial - YouTube](https://www.youtube.com/watch?v=bOS929yCkGE): Learn how to fine-tune OpenAI's GPT LLM to process PDF documents using Langchain and PDF libraries.
- [EasyTune Walkthrough - YouTube](https://www.youtube.com/watch?v=yDYgbKYNk2I) - A walkthrough of fine-tuning LLM with QLoRA on a single GPU using Falcon-7b.
- [Unlocking the Potential of ChatGPT Lessons in Training and Fine ...](https://www.youtube.com/watch?v=Szi0NgmVg7Y) - THE STUDENT presents the instruction fine-tuning and in-context learning of LLMs with symbols.
- [AI News: Creating LLMs without code! - YouTube](https://www.youtube.com/watch?v=MRXj1pAB6cI) - Maya Akim discusses the top 5 LLM fine-tuning use cases you need to know.
- [Top 5 LLM Fine-Tuning Use Cases You Need to Know - YouTube](https://www.youtube.com/watch?v=v1R8uPqjNAU) - An in-depth video highlighting the top 5 LLM fine-tuning use cases with additional links for further exploration.
- [clip2 llm emory - YouTube](https://www.youtube.com/watch?v=UXBA7FKIUso) - Learn how to fine-tune Llama 2 on your own dataset and train an LLM for your specific use case.
- [The EASIEST way to finetune LLAMA-v2 on a local machine! - YouTube](https://www.youtube.com/watch?v=3fsn19OI_C8) - A step-by-step video guide demonstrating the easiest, simplest, and fastest way to fine-tune LLAMA-v2 on your local machine for a custom dataset.
- [Training & Fine-Tuning LLMs: Introduction - YouTube](https://www.youtube.com/watch?v=2QRlvKSzyVw) - An introduction to training and fine-tuning LLMs, including important concepts and the NeurIPS LLM Efficiency Challenge.
- [Fine-tuning LLMs with PEFT and LoRA - YouTube](https://www.youtube.com/watch?v=Us5ZFp16PaU) - A comprehensive video exploring how to use PEFT to fine-tune any decoder-style GPT model, including the basics of LoRA fine-tuning and uploading.
- [Building and Curating Datasets for RLHF and LLM Fine-tuning ...](https://www.youtube.com/watch?v=Ezz_5csCJqI) - Learn about building and curating datasets for RLHF (Reinforcement Learning from Human Feedback) and LLM (Large Language Model) fine-tuning, with sponsorship by Argilla.
- [Fine Tuning LLM (OpenAI GPT) with Custom Data in Python - YouTube](https://www.youtube.com/watch?v=6aOzoJKNLKQ) - Explore how to extend LLM (OpenAI GPT) by fine-tuning it with a custom dataset to provide Q&A, summary, and other ChatGPT-like functions.

## Tools & Software

- [LLaMA Efficient Tuning](https://sourceforge.net/projects/llama-efficient-tuning.mirror/) 🛠️: Easy-to-use LLM fine-tuning framework (LLaMA-2, BLOOM, Falcon).
- [H2O LLM Studio](https://sourceforge.net/projects/h2o-llm-studio.mirror/) 🛠️: Framework and no-code GUI for fine-tuning LLMs.
- [PEFT](https://sourceforge.net/projects/peft.mirror/) 🛠️: Parameter-Efficient Fine-Tuning (PEFT) methods for efficient adaptation of pre-trained language models to downstream applications.
- [ChatGPT-like model](https://sourceforge.net/directory/large-language-models-llm/c/) 🛠️: Run a fast ChatGPT-like model locally on your device.
- [Petals](https://sourceforge.net/projects/petals.mirror/): Run large language models like BLOOM-176B collaboratively, allowing you to load a small part of the model and team up with others for inference or fine-tuning. 🌸
- [NVIDIA NeMo](https://sourceforge.net/directory/large-language-models-llm/linux/): A toolkit for building state-of-the-art conversational AI models and specifically designed for Linux. 🚀
- [H2O LLM Studio](https://sourceforge.net/directory/large-language-models-llm/windows/): A framework and no-code GUI tool for fine-tuning large language models on Windows. 🎛️
- [Ludwig AI](https://sourceforge.net/projects/ludwig-ai.mirror/): A low-code framework for building custom LLMs and other deep neural networks. Easily train state-of-the-art LLMs with a declarative YAML configuration file. 🤖
- [bert4torch](https://sourceforge.net/projects/bert4torch.mirror/): An elegant PyTorch implementation of transformers. Load various open-source large model weights for reasoning and fine-tuning. 🔥
- [Alpaca.cpp](https://sourceforge.net/projects/alpaca-cpp.mirror/): Run a fast ChatGPT-like model locally on your device. A combination of the LLaMA foundation model and an open reproduction of Stanford Alpaca for instruction-tuned fine-tuning. 🦙
- [promptfoo](https://sourceforge.net/projects/promptfoo.mirror/): Evaluate and compare LLM outputs, catch regressions, and improve prompts using automatic evaluations and representative user inputs. 📊

## Conferences & Events

- [ML/AI Conversation: Neuro-Symbolic AI - an Alternative to LLM](https://www.meetup.com/new-york-ai-ml-conversations/events/296127633/) - This meetup will discuss the experience with fine-tuning LLMs and explore neuro-symbolic AI as an alternative.
- [AI Dev Day - Seattle, Mon, Oct 30, 2023, 5:00 PM](https://www.meetup.com/gdgcloudseattle/events/296959536/) - A tech talk on effective LLM observability and fine-tuning opportunities using vector similarity search.
- [DeepLearning.AI Events](https://www.eventbrite.com/o/deeplearningai-19822694300) - A series of events including mitigating LLM hallucinations, fine-tuning LLMs with PyTorch 2.0 and ChatGPT, and AI education programs.
- [AI Dev Day - New York, Thu, Oct 26, 2023, 5:30 PM](https://www.meetup.com/big-data/events/296665127/) - Tech talks on best practices in GenAI applications and using LLMs for real-time, personalized notifications.
- [Chat LLMs & AI Agents - Use Gen AI to Build AI Systems and Agents](https://www.meetup.com/tel-aviv-ai-tech-talks/events/296739549/) - An event focusing on LLMs, AI agents, and chain data, with opportunities for interaction through event chat.
- [NYC AI/LLM/ChatGPT Developers Group](https://www.meetup.com/nyc-llm-talks/) - Regular tech talks/workshops for developers interested in AI, LLMs, ChatGPT, NLP, ML, Data, etc.
- [Leveraging LLMs for Enterprise Data, Tue, Nov 14, 2023, 2:00 PM](https://www.meetup.com/data-science-dojo-new-york/events/296708155/) - Dive into essential LLM strategies tailored for non-public data applications, including prompt engineering and retrieval.
- [Bellevue Applied Machine Learning Meetup](https://www.meetup.com/bellevue-applied-machine-learning-meetup/) - A meetup focusing on applied machine learning techniques and improving the skills of data scientists and ML practitioners.
- [AI & Prompt Engineering Meetup Munich, Do., 5. Okt. 2023, 18:15](https://www.meetup.com/de-DE/ai-prompt-engineering-munich/events/295437909/) - Introduce H2O LLM Studio for fine-tuning LLMs and bring together AI enthusiasts from various backgrounds.
- [Seattle AI/ML/Data Developers Group](https://www.meetup.com/aittg-seattle/) - Tech talks on evaluating LLM agents and learning AI/ML/Data through practice.
- [Data Science Dojo - DC | Meetup](https://www.meetup.com/data-science-dojo-washington-dc/): This is a DC-based meetup group for business professionals interested in teaching, learning, and sharing knowledge and understanding of data science.
- [Find Data Science Events & Groups in Dubai, AE](https://www.meetup.com/find/ae--dubai/data-science/): Discover data science events and groups in Dubai, AE, to connect with people who share your interests.
- [AI Meetup (in-person): Generative AI and LLMs - Halloween Edition](https://www.meetup.com/dc-ai-llms/events/296543682/): Join this AI meetup for a tech talk about generative AI and Large Language Models (LLMs), including open-source tools and best practices.
- [ChatGPT Unleashed: Live Demo and Best Practices for NLP](https://www.meetup.com/data-science-dojo-karachi/events/296977810/): This online event explores fine-tuning hacks for Large Language Models and showcases the practical applications of ChatGPT and LLMs.
- [Find Data Science Events & Groups in Pune, IN](https://www.meetup.com/find/in--pune/data-science/): Explore online or in-person events and groups related to data science in Pune, IN.
- [DC AI/ML/Data Developers Group | Meetup](https://www.meetup.com/aidev-dc/): This group aims to bring together AI enthusiasts in the D.C. area to learn and practice AI technologies, including AI, machine learning, deep learning, and data science.
- [Boston AI/LLMs/ChatGPT Developers Group | Meetup](https://www.meetup.com/bostondeeplearningai/): Join this group in Boston to learn and practice AI technologies like LLMs, ChatGPT, machine learning, deep learning, and data science.
- [Paris NLP | Meetup](https://www.meetup.com/paris-nlp/): This meetup focuses on applications of natural language processing (NLP) in various fields, discussing techniques, research, and applications of both traditional and modern NLP approaches.
- [SF AI/LLMs/ChatGPT Developers Group | Meetup](https://www.meetup.com/san-francisco-ai-llms/): Connect with AI enthusiasts in the San Francisco/Bay area to learn and practice AI tech, including LLMs, ChatGPT, NLP, machine learning, deep learning, and data science.
- [AI meetup (In-person): GenAI and LLMs for Health](https://www.meetup.com/aittg-boston/events/296567040/): Attend this tech talk about the application of LLMs in healthcare and learn about quick wins in using LLMs for health-related tasks.

## Slides & Presentations

- [Fine tuning large LMs](https://www.slideshare.net/SylvainGugger/fine-tuning-large-lms-243430468): Presentation discussing the process of fine-tuning large language models like GPT, BERT, and RoBERTa.
- [LLaMa 2.pptx](https://www.slideshare.net/RkRahul16/llama-2pptx): Slides introducing LLaMa 2, a powerful large language model successor developed by Meta AI.
- [LLM.pdf](https://www.slideshare.net/MedBelatrach/llmpdf-261239806): Presentation exploring the role of Transformers in NLP, from BERT to GPT-3.
- [Large Language Models Bootcamp](https://www.slideshare.net/DataScienceDojo/large-language-models-bootcamp): Bootcamp slides covering various aspects of large language models, including training from scratch and fine-tuning.
- [The LHC Explained by CNN](https://www.slideshare.net/hijiki_s/the-lhc-explained-by-cnn): Slides explaining the LHC (Large Hadron Collider) using CNN and fine-tuning image models.
- [Using Large Language Models in 10 Lines of Code](https://www.slideshare.net/GautierMarti/using-large-language-models-in-10-lines-of-code): Presentation demonstrating how to use large language models in just 10 lines of code.
- [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention.pdf](https://www.slideshare.net/jacksonChen22/llamaadapter-efficient-finetuning-of-language-models-with-zeroinit-attentionpdf): Slides discussing LLaMA-Adapter, an efficient technique for fine-tuning language models with zero-init attention.
- [Intro to LLMs](https://www.slideshare.net/LoicMerckel/intro-to-llms): Presentation providing an introduction to large language models, including base models and fine-tuning with prompt-completion pairs.
- [LLM Fine-Tuning (東大松尾研LLM講座 Day5資料) - Speaker Deck](https://speakerdeck.com/schulta/llm-fine-tuning-dong-da-song-wei-yan-llmjiang-zuo-day5zi-liao): Slides used for a lecture on fine-tuning large language models, specifically for the 東大松尾研サマースクール2023.
- [Automate your Job and Business with ChatGPT #3](https://pt.slideshare.net/AnantCorp/automate-your-job-and-business-with-chatgpt-3-fundamentals-of-llmgpt): Presentation discussing the fundamentals of ChatGPT and its applications for job automation and business tasks.
- [Unlocking the Power of Generative AI An Executive's Guide.pdf](https://www.slideshare.net/PremNaraindas1/unlocking-the-power-of-generative-ai-an-executives-guidepdf) - A guide that explains the process of fine-tuning Large Language Models (LLMs) to tailor them to an organization's needs.
- [Fine tune and deploy Hugging Face NLP models | PPT](https://www.slideshare.net/ovhcom/pres-hugging-facefinetuning) - A presentation that provides insights on how to build and deploy LLM models using Hugging Face NLP.
- [大規模言語モデル時代のHuman-in-the-Loop機械学習 - Speaker Deck](https://speakerdeck.com/yukinobaba/human-in-the-loop-ml-llm) - A slide deck discussing the process of fine-tuning Language Models to find agreement among humans with diverse preferences.
- [AI and ML Series - Introduction to Generative AI and LLMs | PPT](https://www.slideshare.net/DianaGray10/ai-and-ml-series-introduction-to-generative-ai-and-llms-session-1) - A presentation introducing Generative AI and LLMs, including their usage in specific applications.
- [Retrieval Augmented Generation in Practice: Scalable GenAI ...](https://www.slideshare.net/cmihai/retrieval-augmented-generation-in-practice-scalable-genai-platforms-with-k8s-langchain-huggingface-and-vector) - A presentation discussing use cases for Generative AI, limitations of Large Language Models, and the use of Retrieval Augmented Generation (RAG) and fine-tuning techniques.
- [LLM presentation final | PPT](https://www.slideshare.net/RuthGriffin3/llm-presentation-final) - A presentation covering the Child & Family Agency Act 2013 and the Best Interest Principle in the context of LLMs.
- [LLM Paradigm Adaptations in Recommender Systems.pdf](https://www.slideshare.net/NagaBathula1/llm-paradigm-adaptations-in-recommender-systemspdf) - A PDF explaining the fine-tuning process and objective adaptations in LLM-based recommender systems.
- [Conversational AI with Transformer Models | PPT](https://www.slideshare.net/databricks/conversational-ai-with-transformer-models) - A presentation highlighting the use of Transformer Models in Conversational AI applications.
- [Llama-index | PPT](https://pt.slideshare.net/Denis973830/llamaindex) - A presentation on the rise of LLMs and building LLM-powered applications.
- [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention.pdf](https://www.slideshare.net/jacksonChen22/llamaadapter-efficient-finetuning-of-language-models-with-zeroinit-attentionpdf) - A PDF discussing the efficient fine-tuning of Language Models with zero-init attention using LLaMA.

## Podcasts

- [Practical AI: Machine Learning, Data Science](https://open.spotify.com/show/1LaCr5TFAgYPK5qHjP3XDp) 🎧 - Making artificial intelligence practical, productive, and accessible to everyone. Engage in lively discussions about AI, machine learning, deep learning, neural networks, and more. Accessible insights and real-world scenarios for both beginners and seasoned practitioners.
- [Gradient Dissent: Exploring Machine Learning, AI, Deep Learning](https://podcasts.apple.com/us/podcast/gradient-dissent-exploring-machine-learning-ai-deep/id1504567418) 🎧 - Go behind the scenes to learn from industry leaders about how they are implementing deep learning in real-world scenarios. Gain insights into the machine learning industry and stay updated with the latest trends.
- [Weaviate Podcast](https://open.spotify.com/show/4TlG6dnrWYdgN2YHpoSnM7) 🎧 - Join Connor Shorten for the Weaviate Podcast series, featuring interviews with experts and discussions on AI-related topics.
- [Latent Space: The AI Engineer Podcast — CodeGen, Agents, Computer Vision, Data Science, AI UX and all things Software 3.0](https://podcasts.apple.com/in/podcast/latent-space-the-ai-engineer-podcast-codegen-agents/id1674008350) 🎧 - Dive into the world of AI engineering, covering topics like code generation, computer vision, data science, and the latest advancements in AI UX.
- [Unsupervised Learning](https://podcasts.apple.com/il/podcast/unsupervised-learning/id1672188924) 🎧 - Gain insights into the rapidly developing AI landscape and its impact on businesses and the world. Explore discussions on LLM applications, trends, and disrupting technologies.
- [The TWIML AI Podcast (formerly This Week in Machine Learning)](https://podcasts.apple.com/no/podcast/the-twiml-ai-podcast-formerly-this-week-in-machine/id1116303051) 🎧 - Dive deep into fine-tuning approaches used in AI, LLM capabilities and limitations, and learn from experts in the field.
- [AI and the Future of Work on Apple Podcasts](https://podcasts.apple.com/us/podcast/ai-and-the-future-of-work/id1476885647): A podcast hosted by SC Moatti discussing the impact of AI on the future of work.
- [Practical AI: Machine Learning, Data Science: Fine-tuning vs RAG](https://podcasts.apple.com/us/podcast/fine-tuning-vs-rag/id1406537385?i=1000626951912): This episode explores the comparison between fine-tuning and retrieval augmented generation in machine learning and data science.
- [Unsupervised Learning on Apple Podcasts](https://podcasts.apple.com/fi/podcast/unsupervised-learning/id1672188924): Episode 20 features an interview with Anthropic CEO Dario Amodei on the future of AGI and AI.
- [Papers Read on AI | Podcast on Spotify](https://open.spotify.com/show/2w8DRieJhMGFSTUhnsTVrw): This podcast keeps you updated with the latest trends and best performing architectures in the field of computer science.
- [This Day in AI Podcast on Apple Podcasts](https://podcasts.apple.com/in/podcast/this-day-in-ai-podcast/id1671087656): Covering various AI-related topics, this podcast offers exciting insights into the world of AI.
- [All About Evaluating LLM Applications // Shahul Es // #179 MLOps](https://player.fm/series/mlopscommunity/all-about-evaluating-llm-applications-shahul-es-mlops-podcast-179): In this episode, Shahul Es shares his expertise on evaluation in open source models, including insights on debugging, troubleshooting, and benchmarks.
- [AI Daily on Apple Podcasts](https://podcasts.apple.com/us/podcast/ai-daily/id1686002118): Hosted by Conner, Ethan, and Farb, this podcast explores fascinating AI-related stories.
- [Yannic Kilcher Videos (Audio Only) | Podcast on Spotify](https://open.spotify.com/show/6cHS7bXU2JPLTgjA0z0xNz): Yannic Kilcher discusses machine learning research papers, programming, and the broader impact of AI in society.
- [LessWrong Curated Podcast | Podcast on Spotify](https://open.spotify.com/show/7vqBzO0ejqiLiXyTECEeBY): Audio version of the posts shared in the LessWrong Curated newsletter.
- [SAI: The Security and AI Podcast on Apple Podcasts](https://podcasts.apple.com/il/podcast/sai-the-security-and-ai-podcast/id1690378369): An episode focused on OpenAI's cybersecurity grant program.

---

This initial version of the Awesome List was generated with the help of the [Awesome List Generator](https://github.com/alialsaeedi19/GPT-Awesome-List-Maker). It's an open-source Python package that uses the power of GPT models to automatically curate and generate starting points for resource lists related to a specific topic. 
