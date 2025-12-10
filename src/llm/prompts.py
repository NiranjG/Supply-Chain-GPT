"""
Prompt templates for LLM orchestration
"""

from typing import List, Dict, Any, Optional


class PromptTemplates:
    """Prompt templates for SupplyChainGPT"""

    SYSTEM_PROMPT = """You are SupplyChainGPT, an AI assistant specialized in supply chain management,
inventory planning, and warehouse operations. Your role is to:

1. Answer questions based on retrieved context from internal documents
2. Provide actionable recommendations for inventory management
3. Explain forecasts and safety stock calculations
4. Help users understand supply chain policies and procedures

Guidelines:
- Be concise and factual
- Always cite your sources using [Source: document_title] format
- Quote small excerpts (<=50 words) when relevant
- If information is insufficient, clearly state what's missing
- Never make up facts not supported by the context
- When providing numbers, always mention the source
- If asked about forecasts, explain the methodology used

Response Format:
- Start with a direct answer to the question
- Provide supporting details with citations
- End with actionable recommendations if applicable
"""

    QUERY_WITH_CONTEXT_TEMPLATE = """Based on the following context, answer the user's question.

CONTEXT:
{context}

{forecast_context}

USER QUESTION: {question}

Instructions:
1. Answer based ONLY on the provided context
2. Cite sources using [Source: title] format
3. If the context doesn't contain enough information, say so
4. Be specific and actionable

ANSWER:"""

    CONTEXT_CHUNK_TEMPLATE = """[SOURCE {index}]
Document: {doc_title}
Type: {doc_type}
Relevance Score: {score:.2f}

Content:
{content}

---"""

    FORECAST_CONTEXT_TEMPLATE = """
FORECAST DATA:
SKU: {sku_id}
Warehouse: {warehouse_id}
Model Used: {model}
Forecast Period: {periods} days

Predicted Demand Summary:
- Average Daily: {avg_demand:.1f} units
- Total: {total_demand:.0f} units
- Peak: {max_demand:.1f} units on {peak_date}

{safety_stock_info}
"""

    SAFETY_STOCK_TEMPLATE = """
Safety Stock Analysis:
- Recommended Safety Stock: {safety_stock:.0f} units
- Reorder Point: {reorder_point:.0f} units
- Service Level: {service_level:.0%}
"""

    NO_CONTEXT_RESPONSE = """I don't have enough information in the available documents to answer this question accurately.

To help you better, please:
1. Upload relevant documents (SOPs, reports, contracts)
2. Be more specific about the SKU, warehouse, or time period
3. Check if the information might be in a different document type

What I can help with:
- Inventory policies and procedures
- Demand forecasting for specific SKUs
- Safety stock calculations
- Supplier and contract information
"""

    GUARDRAIL_PROMPT = """Review the following response for:
1. Factual accuracy based on cited sources
2. Potential hallucinations or unsupported claims
3. Appropriate citations for all facts
4. Safety and compliance issues

Response to review:
{response}

Sources available:
{sources}

Issues found (if any):"""

    @classmethod
    def format_context(
        cls,
        chunks: List[Dict[str, Any]],
        max_chunks: int = 5
    ) -> str:
        """Format retrieved chunks into context string"""
        context_parts = []

        for i, chunk in enumerate(chunks[:max_chunks], 1):
            metadata = chunk.get("metadata", {})
            context_parts.append(cls.CONTEXT_CHUNK_TEMPLATE.format(
                index=i,
                doc_title=metadata.get("doc_title", "Unknown"),
                doc_type=metadata.get("doc_type", "document"),
                score=chunk.get("similarity", chunk.get("combined_score", 0)),
                content=chunk.get("chunk_text", "")[:1500]  # Limit chunk size
            ))

        return "\n".join(context_parts)

    @classmethod
    def format_forecast_context(
        cls,
        forecast_result: Optional[Dict[str, Any]],
        safety_stock_result: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format forecast data into context string"""
        if not forecast_result:
            return ""

        forecast = forecast_result.get("forecast", [])
        if not forecast:
            return ""

        import numpy as np
        demands = [f["predicted_demand"] for f in forecast]

        peak_idx = np.argmax(demands)
        peak_date = forecast[peak_idx]["date"]

        safety_info = ""
        if safety_stock_result:
            safety_info = cls.SAFETY_STOCK_TEMPLATE.format(
                safety_stock=safety_stock_result.get("safety_stock", 0),
                reorder_point=safety_stock_result.get("reorder_point", 0),
                service_level=safety_stock_result.get("service_level", 0.95)
            )

        return cls.FORECAST_CONTEXT_TEMPLATE.format(
            sku_id=forecast_result.get("sku_id", "Unknown"),
            warehouse_id=forecast_result.get("warehouse_id", "All"),
            model=forecast_result.get("model_used", "Unknown"),
            periods=len(forecast),
            avg_demand=np.mean(demands),
            total_demand=np.sum(demands),
            max_demand=np.max(demands),
            peak_date=peak_date,
            safety_stock_info=safety_info
        )

    @classmethod
    def build_query_prompt(
        cls,
        question: str,
        chunks: List[Dict[str, Any]],
        forecast_result: Optional[Dict[str, Any]] = None,
        safety_stock_result: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build complete prompt for query answering"""
        context = cls.format_context(chunks)
        forecast_context = cls.format_forecast_context(forecast_result, safety_stock_result)

        if not context and not forecast_context:
            return cls.NO_CONTEXT_RESPONSE

        return cls.QUERY_WITH_CONTEXT_TEMPLATE.format(
            context=context,
            forecast_context=forecast_context,
            question=question
        )

    @classmethod
    def extract_citations(cls, response: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citations from response and match to source chunks"""
        import re

        citations = []
        seen_titles = set()

        # Find all [Source: ...] patterns
        citation_pattern = r"\[Source:\s*([^\]]+)\]"
        matches = re.findall(citation_pattern, response)

        for match in matches:
            title = match.strip()
            if title in seen_titles:
                continue
            seen_titles.add(title)

            # Find matching chunk
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                doc_title = metadata.get("doc_title", "")

                if title.lower() in doc_title.lower() or doc_title.lower() in title.lower():
                    citations.append({
                        "doc_title": doc_title,
                        "source_uri": metadata.get("source_uri", ""),
                        "chunk_text": chunk.get("chunk_text", "")[:200],
                        "relevance_score": chunk.get("similarity", chunk.get("combined_score", 0))
                    })
                    break

        return citations
