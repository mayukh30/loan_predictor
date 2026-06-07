import os
import pickle
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class AdviceDocument:
    id: str
    title: str
    category: str
    content: str
    action: str
    impact: str


ADVICE_CATALOG: Sequence[AdviceDocument] = (
    AdviceDocument(
        id="lower_loan_amount",
        title="Reduce the requested loan amount",
        category="Loan Size",
        content="Ask for a smaller loan amount when your income and repayment history are borderline. Lower principal amounts generally reduce lender risk and improve approval odds.",
        action="Try reducing the requested amount or split the purchase into a smaller first loan.",
        impact="High",
    ),
    AdviceDocument(
        id="improve_credit_history",
        title="Strengthen your credit history",
        category="Credit",
        content="A clean repayment record, low utilization, and no missed payments can materially improve loan approval decisions. Credit history is one of the strongest risk signals.",
        action="Pay down existing dues, correct credit report errors, and avoid new delinquent accounts.",
        impact="High",
    ),
    AdviceDocument(
        id="add_coapplicant",
        title="Add a strong co-applicant",
        category="Income",
        content="A co-applicant with stable income and good credit can offset a weak profile. Lenders often consider combined repayment capacity and history.",
        action="Add a salaried spouse or family member with documented income and a stable credit record.",
        impact="High",
    ),
    AdviceDocument(
        id="increase_documented_income",
        title="Increase documented income",
        category="Income",
        content="Proof of higher and stable income improves affordability checks. Lenders prefer income that is consistent and easy to verify.",
        action="Provide salary slips, tax returns, bank statements, or formal income proof before reapplying.",
        impact="High",
    ),
    AdviceDocument(
        id="add_collateral",
        title="Offer collateral or security",
        category="Risk Mitigation",
        content="Collateral reduces lender exposure and can help borderline applications move closer to approval.",
        action="Ask the lender about secured-loan options or assets that may qualify as collateral.",
        impact="Medium",
    ),
    AdviceDocument(
        id="stabilize_self_employment",
        title="Document self-employment income",
        category="Income Stability",
        content="Self-employed borrowers can improve odds by showing more predictable income through bank trails and audited statements.",
        action="Prepare audited statements, GST/tax filings, and consistent bank inflows for the last 6 to 12 months.",
        impact="Medium",
    ),
    AdviceDocument(
        id="optimize_loan_term",
        title="Choose a repayment term that fits your cash flow",
        category="Repayment Plan",
        content="A repayment term aligned with your monthly budget makes your application easier to justify. The goal is to show that the EMI is sustainable.",
        action="Adjust the loan term so the monthly installment is comfortably covered by your verified income.",
        impact="Medium",
    ),
    AdviceDocument(
        id="improve_property_signal",
        title="Strengthen the property profile",
        category="Collateral Profile",
        content="Lenders may see urban or semiurban property as easier to value and resell. A clearer property profile can reduce uncertainty.",
        action="If possible, include property documents that are easy to verify and discuss alternate collateral with the lender.",
        impact="Low",
    ),
    AdviceDocument(
        id="reapply_after_improvements",
        title="Reapply after targeted improvements",
        category="Strategy",
        content="If the current application is weak, it is often better to improve the most important factors first and then submit a stronger profile.",
        action="Fix the top weak areas, wait for updated documents, then submit a fresh application.",
        impact="High",
    ),
)


class LoanAdviceVectorStore:
    def __init__(self, persist_path: str):
        self.persist_path = persist_path
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.documents: List[AdviceDocument] = []
        self._load_or_build()

    def _load_or_build(self) -> None:
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "rb") as handle:
                payload = pickle.load(handle)
            self.vectorizer = payload["vectorizer"]
            self.matrix = payload["matrix"]
            self.documents = payload["documents"]
            return

        self.documents = list(ADVICE_CATALOG)
        texts = [self._document_text(document) for document in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)

        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "wb") as handle:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "matrix": self.matrix,
                    "documents": self.documents,
                },
                handle,
            )

    @staticmethod
    def _document_text(document: AdviceDocument) -> str:
        return " ".join([document.title, document.category, document.content, document.action, document.impact])

    def query(self, text: str, top_k: int = 4) -> List[dict]:
        if not text.strip() or self.vectorizer is None or self.matrix is None:
            return []

        query_vector = self.vectorizer.transform([text])
        similarities = (self.matrix @ query_vector.T).toarray().ravel()
        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for index in ranked_indices:
            score = float(similarities[index])
            if score <= 0:
                continue

            document = self.documents[index]
            results.append(
                {
                    "id": document.id,
                    "title": document.title,
                    "category": document.category,
                    "content": document.content,
                    "action": document.action,
                    "impact": document.impact,
                    "similarity": round(score, 4),
                }
            )

        return results


def build_advice_query(application: dict, rejection_reasons: Sequence[dict]) -> str:
    parts = [
        f"income {application.get('ApplicantIncome', 0)}",
        f"coapplicant income {application.get('CoapplicantIncome', 0)}",
        f"loan amount {application.get('LoanAmount', 0)}",
        f"loan term {application.get('Loan_Amount_Term', 0)}",
        f"credit history {application.get('Credit_History', 0)}",
        f"self employed {application.get('Self_Employed', 0)}",
        f"dependents {application.get('Dependents', 0)}",
    ]

    for reason in rejection_reasons:
        parts.append(reason.get("feature", ""))
        parts.append(reason.get("explanation", ""))

    return " ".join(str(part) for part in parts if part)


def summarize_recommendations(results: Sequence[dict]) -> List[str]:
    return [f"{item['title']}: {item['action']}" for item in results]
