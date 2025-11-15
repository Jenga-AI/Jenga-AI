# Hackathon Abstract: AI for National Prosperity

## Title: Jenga-AI: A Unified Multi-Task Framework for National Security and Governance in Kenya

**Challenge Tracks Addressed:** 
*   Threat Detection & Prevention
*   Cyber Intelligence
*   Governance & Public Policy

---

## Problem Statement

Kenya, like many African nations, faces complex security and governance challenges that are increasingly playing out in the digital sphere. These include the rapid spread of misinformation and hate speech in local languages like Swahili and Sheng, the coordination of organized crime, and the need to analyze public sentiment on policy issues. Existing AI and NLP models, predominantly trained on Western data, are ill-suited for this landscape. They fail to grasp the unique linguistic nuances, cultural contexts, and code-switching prevalent in Kenyan communication.

Furthermore, addressing these issues requires tackling multiple problems at once—a text may contain hate speech (a classification task) while also mentioning specific actors and locations (a Named Entity Recognition task). For most Kenyan organizations—from startups to government agencies—the cost and complexity of developing, training, and deploying separate, specialized AI models for each individual task is prohibitively high. This technological barrier prevents the widespread adoption of AI for enhancing national security and strengthening public service delivery, leaving a critical gap in our intelligence and governance capabilities.

---

## Our Solution

Jenga-AI directly addresses this challenge with a highly efficient, open-source, multi-task learning framework. Our solution allows a **single, unified model** to be trained simultaneously on multiple, distinct NLP tasks. By leveraging a shared transformer-based encoder and innovative attention fusion mechanisms, Jenga-AI can, for instance, analyze a piece of text to **concurrently** (1) classify it as a potential threat, (2) extract the names of involved organizations, and (3) categorize its relevance to a specific public policy.

This approach dramatically lowers the computational and financial barriers to entry, enabling local organizations to build and deploy sophisticated, context-aware AI solutions that are tailored to the Kenyan environment. For this hackathon, our MVP demonstrates a model trained on synthetic data to simultaneously perform threat classification and named entity recognition, providing a powerful tool for real-time intelligence that is both affordable and scalable. Jenga-AI is not just a model; it's a force multiplier for democratizing AI-driven security and governance across Kenya.
