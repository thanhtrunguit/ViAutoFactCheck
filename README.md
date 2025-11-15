<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>


# **ViAutoFactCheck: Integrating Context Retrieval and Machine Reading Comprehension for Automated Fact Verification**  

### **Authors**
- Ngô Thành Trung  
- Đinh Nhật Trường  

### **Supervisors**
- ThS. Huỳnh Văn Tín  
- TS. Nguyễn Văn Kiệt  

---

### **Background & Motivation**

In today's information-rich world, the rapid spread of news, social media posts, and online content has made it increasingly difficult to discern true information from false or misleading claims. **Fact-checking** is essential to ensure the credibility of information, prevent misinformation, and support informed decision-making in society. 

This challenge motivates the need for **Automated Fact-Checking systems**, which leverage natural language processing (NLP) and machine learning techniques to:  
- **Retrieve relevant evidence** from large knowledge bases efficiently.  
- **Understand and extract information** using machine reading comprehension models.  
- **Verify claims automatically** by comparing statements with retrieved evidence.

### **Project Overview**

**ViAutoFactCheck** is an end-to-end automated fact verification pipeline that integrates three key components to verify claims efficiently and accurately:

1. **Information Retrieval (IR)**  
   - Retrieves relevant documents or context from a large knowledge base based on the input claim.  
   - Ensures the system has the most relevant evidence before verification.

2. **Machine Reading Comprehension (MRC)**  
   - Extracts precise information or answers from the retrieved context.  
   - Understands the content to identify evidence supporting or refuting the claim.
   - **Leverages sliding window splitting** to handle long contexts efficiently, ensuring no important information is missed.

3. **Text Classification**  
   - Compares the extracted evidence with the original claim.  
   - Assigns a label such as *Supported*, *Refuted*, or *Not Enough Information*.

These components work together in a seamless, automated pipeline to perform **robust fact verification** without manual intervention.

We perform experiments on various number of State-of-the-art (SOTA) models such as [CafeBERT](https://huggingface.co/uitnlp/CafeBERT), [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-large), [InfoXLM](https://huggingface.co/microsoft/infoxlm-large), [Vi-MRC](https://huggingface.co/nguyenvulebinh/vi-mrc-large), v.v and tested the pipeline on [ViWikiFC](https://arxiv.org/abs/2405.07615).

---

### **Information Retrieval (IR)**

The Information Retrieval component is responsible for finding relevant context for a given claim from a large knowledge base.  
ViAutoFactCheck uses a **two-stage retrieval strategy** for optimal performance:

1. **Bi-Encoder Retrieval**  
   - Uses the [`bkai-foundation-models/vietnamese-bi-encoder`](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder) to encode claims and documents into embeddings.  
   - Retrieves the **top 50 most relevant documents** efficiently using cosine similarity.

2. **Cross-Encoder Re-Ranking**  
   - Uses the [`itdainb/PhoRanker`](https://huggingface.co/itdainb/PhoRanker) cross-encoder to score the relevance of candidate documents with respect to the claim.  
   - Selects the **top 1–5 documents** for downstream MRC processing, ensuring high-quality evidence.

This **two-stage IR approach** balances efficiency and accuracy, leveraging the speed of bi-encoders with the precision of cross-encoders.

---

### **2. Machine Reading Comprehension (MRC)**

The MRC component is responsible for extracting precise evidence or answers from the retrieved context. This is also known as **span-based extraction**, where the model identifies and extracts a specific span of text from the context.


We trained and evaluated several state-of-the-art models:  
- [CafeBERT](https://huggingface.co/uitnlp/CafeBERT)  
- [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-large)  
- [InfoXLM](https://huggingface.co/microsoft/infoxlm-large)  
- [Vi-MRC](https://huggingface.co/nguyenvulebinh/vi-mrc-large)  

To handle long contexts that exceed the model's maximum input length, we experimented with two strategies:  
1. **Hard Split Context** – Split the context strictly based on the **length of the stripped context**
- *Stripped context* refers to the context text after removing unnecessary characters, extra spaces, or formatting tokens, keeping only the clean textual content for accurate token length calculation.  
2. **Sliding Window** – Split the context with an **overlap of 100 tokens**, ensuring no important information is lost.  
- *Sliding window* refers to moving a fixed-length window over the stripped context so that consecutive segments share 100 tokens, allowing the model to see context spanning across splits.

> All token lengths are calculated using the respective model's tokenizer.

This setup allows the MRC module to extract accurate evidence even from very long documents, balancing context coverage and model capacity.

---

### **3. Text Classification**

The Text Classification component verifies claims using the evidence extracted from the MRC module.  

We trained [CafeBERT](https://huggingface.co/uitnlp/CafeBERT), [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-large), and [InfoXLM](https://huggingface.co/microsoft/infoxlm-large) to predict labels based on the combination of the claim and the extracted evidence. 

The possible labels are:  
- **Supported** – Evidence confirms the claim.  
- **Refuted** – Evidence contradicts the claim.  
- **Not Enough Information (NEI)** – Retrieved context is insufficient to verify the claim.

---

### **Dataset**

For training and evaluating ViAutoFactCheck, we use the [ViWikiFC](https://arxiv.org/abs/2405.07615), a Vietnamese fact-checking benchmark.  

|           | ViWikiFC |
|-----------|---------:|
| **Train** | 16,738   |
| **Dev**   | 2,090    |
| **Test**  | 2,091    |

Each claim is paired with relevant context and evidence, and labeled as **Supported**, **Refuted**, or **Not Enough Information (NEI)**.  
This dataset is suitable for training all three components of the pipeline:  
- **Information Retrieval (IR)** – to retrieve relevant context, from claim-context pairs.
- **Machine Reading Comprehension (MRC)** – to extract precise evidence with claim-context and evidence.  
- **Text Classification** – to predict the final claim label with claim-evidence.

---

## **Evaluation**

We evaluate **ViAutoFactCheck** at both the component level and the end-to-end pipeline level using standard metrics for retrieval, reading comprehension, classification, and claim verification.

### **1. Information Retrieval (IR)**

- **MRR@10 (Mean Reciprocal Rank)** – Measures the quality of general retrieval by evaluating the rank of the first relevant document among the top 10 retrieved documents.  
- **ACC@1 and ACC@2** – Measure the accuracy of hard, precise retrieval by checking whether the top 1 or top 2 retrieved documents contain the actual relevant evidence. In real-world applications, retrieving more documents increases the system's runtime, so limiting to the top 1–2 strikes a balance between accuracy and efficiency.


### **2. Machine Reading Comprehension (MRC)**

- **Exact Match (EM)** – Evaluates the ability of the MRC module to extract precise evidence by measuring the percentage of predictions that exactly match the reference answer span.

### **3. Text Classification (CLS)**

- **Accuracy** – Measures the percentage of claims that are correctly classified as *Supported*, *Refuted*, or *Not Enough Information*.  
- **F1 Score** – Captures the balance between precision and recall for classification, especially useful when labels are imbalanced.

### **4. End-to-End Pipeline**

The full pipeline is evaluated using claim verification metrics:  
- **Strict Accuracy (Strict Acc)** – Counts claims as correct only if both the predicted label and the retrieved evidence are correct.  
- **Evidence Retrieval Accuracy (ER Acc)** – Measures the ability to retrieve correct supporting evidence, regardless of the predicted label.  
- **Verification Classification Accuracy (VC Acc)** – Measures the accuracy of the predicted claim label, regardless of whether the retrieved evidence is exactly correct.

> This evaluation framework ensures that each module performs well individually and that the full pipeline reliably verifies claims using retrieved evidence.
