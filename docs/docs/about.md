# About Jenga-AI

Jenga-AI is more than a software library; it's a response to a critical challenge and a tool for empowerment. This page outlines the problem we are solving, our technical solution, and our vision for the future of AI in Africa.

---

## The Problem: A Gap in Our Digital Landscape

Kenya, like many African nations, faces complex security and governance challenges that are increasingly playing out in the digital sphere. These include:
- The rapid spread of misinformation and hate speech in local languages like **Swahili** and **Sheng**.
- The coordination of organized crime through digital channels.
- The need to analyze public sentiment on critical policy issues.

Existing AI and NLP models, which are predominantly trained on Western data, are ill-suited for this landscape. They fail to grasp the unique linguistic nuances, cultural contexts, and code-switching prevalent in Kenyan communication.

Furthermore, addressing these issues requires tackling multiple problems at once—a text may contain hate speech (a classification task) while also mentioning specific actors and locations (a Named Entity Recognition task). For most local organizations, the cost and complexity of developing, training, and deploying separate, specialized AI models for each task is prohibitively high.

This technological barrier prevents the widespread adoption of AI for enhancing national security and strengthening public service delivery, leaving a critical gap in our intelligence and governance capabilities.

---

## Our Solution: A Unified, Multi-Task Framework

Jenga-AI directly addresses this challenge with a highly efficient, open-source, **multi-task learning framework**.

Our solution allows a **single, unified model** with a shared encoder to be trained simultaneously on multiple, distinct NLP tasks. By leveraging a shared transformer architecture and innovative attention fusion mechanisms, Jenga-AI can, for instance, analyze a piece of text to **concurrently**:
1.  Classify it as a potential threat.
2.  Extract the names of involved organizations.
3.  Categorize its relevance to a specific public policy.

This approach dramatically lowers the computational and financial barriers to entry, enabling local organizations to build and deploy sophisticated, context-aware AI solutions that are tailored to the Kenyan environment.

### Key Capabilities

Jenga-AI is designed to be a versatile platform for a wide array of NLP challenges. Its adaptable, config-driven nature makes it suitable for:

- **Named Entity Recognition (NER):** Identifying people, places, and organizations.
- **Text Classification:** For tasks like sentiment analysis or threat detection.
- **Multi-Label Classification:** Assigning multiple tags or categories to a piece of text.
- **Question Answering (QA):** Building systems that can answer questions based on a context.
- **QA Quality Control (QA-QC):** Evaluating the quality of question-answer pairs.
- **Translation:** Fine-tuning models for machine translation between languages.
- **LLM Finetuning:** Adapting large language models for specific domains or tasks.

---

## Our Vision: Democratizing AI for National Prosperity

Jenga-AI is not just a model; it's a **force multiplier for democratizing AI-driven security and governance across Kenya**. We aim to provide the tools and the knowledge necessary for African developers and researchers to solve African problems. By making advanced NLP accessible, adaptable, and affordable, we can collectively build a safer and more prosperous digital future.