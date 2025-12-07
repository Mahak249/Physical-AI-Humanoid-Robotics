<!--
Sync Impact Report:
Version change: 0.0.0 → 1.0.0
List of modified principles: All principles updated with specific descriptions
Added sections: Technical Stack & Infrastructure, Development & Operations Guidelines
Removed sections: None
Templates requiring updates:
- .specify/templates/plan-template.md: ⚠ pending
- .specify/templates/spec-template.md: ⚠ pending
- .specify/templates/tasks-template.md: ⚠ pending
- .specify/templates/commands/*.md: ⚠ pending
Follow-up TODOs:
- TODO(RATIFICATION_DATE): If this is not the initial ratification, update with original date.
-->
# Physical AI & Humanoid Robotics Textbook Hackathon Constitution

## Core Principles

### Modular Docusaurus Textbook
The textbook will be built on Docusaurus, with chapters organized as individual markdown files, facilitating modular content management and easy updates.

### Secure User Authentication & Profiling
User authentication and profiling will be handled via Better Auth to enable personalized content delivery based on hardware and software background.

### Multilingual Support
The platform will include multilingual capabilities, with an on-demand toggle for Urdu translation.

### RAG Chatbot Integration
An embedded RAG chatbot will be built using OpenAI Agents/ChatKit, backed by FastAPI, Neon Serverless Postgres, and a Qdrant vector database for answering questions from both full content and user-selected text.

### Full CI/CD Pipeline
A comprehensive CI/CD pipeline will be implemented for automated embedding updates, testing, and deployment processes.

### Scalable and Extensible Architecture
The architecture will be designed for scalability to support future integration of subagents and skills, promoting reusability and extensibility.

## Technical Stack & Infrastructure

The core technologies include Docusaurus for the frontend, Better Auth for user management, OpenAI Agents/ChatKit for AI components, FastAPI for the backend API, Neon Serverless Postgres for relational data, and Qdrant for vector embeddings. Deployment will leverage a full CI/CD pipeline.

## Development & Operations Guidelines

Development will follow a spec-driven approach. All features will be thoroughly tested. Continuous integration and continuous deployment will ensure regular updates and robust performance. Operational readiness will focus on observability through logging, metrics, and tracing, with defined alerting and runbook procedures.

## Governance

This constitution defines the fundamental principles and guidelines for the 'Physical AI & Humanoid Robotics Textbook Hackathon' project. All development, design, and operational activities must adhere to these principles. Amendments to this constitution require a documented proposal, review, and approval from the core team, along with a clear migration plan for any affected systems. Compliance with these rules will be regularly reviewed, and any deviations must be formally justified and approved.

**Version**: 1.0.0 | **Ratified**: 2025-12-06 | **Last Amended**: 2025-12-06
