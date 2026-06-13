"""
Finance Calculator — Deterministic loan and affordability calculations.

The LLM is NOT used for arithmetic. All numbers here are computed exactly
using standard financial formulas. The LLM only formats/explains the output.

Formulas:
  Monthly payment (PMT):
    M = P * [r(1+r)^n] / [(1+r)^n - 1]
    where P = principal, r = monthly rate, n = number of months

  Max affordable loan (given monthly income and DTI):
    P = M * [(1+r)^n - 1] / [r(1+r)^n]
    where M = max_monthly_payment = monthly_income * dti_ratio
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LoanPlan:
    """Result of a loan calculation."""
    principal_vnd: float           # Khoản vay (VND)
    annual_rate_pct: float         # Lãi suất/năm (%)
    term_months: int               # Số tháng vay
    monthly_payment_vnd: float     # Trả hàng tháng (VND)
    total_payment_vnd: float       # Tổng trả (VND)
    total_interest_vnd: float      # Tổng lãi (VND)
    # Down payment context (optional)
    property_price_vnd: Optional[float] = None
    down_payment_vnd: Optional[float] = None
    down_payment_pct: Optional[float] = None

    def summary_text(self) -> str:
        """Vietnamese summary for LLM context injection."""
        def fmt(v):
            b = v / 1_000_000_000
            m = v / 1_000_000
            return f"{b:.2f} tỷ" if b >= 1 else f"{m:.0f} triệu"

        lines = [
            "=== KẾT QUẢ TÍNH TOÁN TÀI CHÍNH (chính xác, không phải ước tính) ===",
            f"  Khoản vay:          {fmt(self.principal_vnd)}",
            f"  Lãi suất/năm:       {self.annual_rate_pct:.1f}%",
            f"  Thời hạn:           {self.term_months // 12} năm ({self.term_months} tháng)",
            f"  Trả hàng tháng:     {fmt(self.monthly_payment_vnd)}",
            f"  Tổng số tiền trả:   {fmt(self.total_payment_vnd)}",
            f"  Tổng lãi phải trả:  {fmt(self.total_interest_vnd)}",
        ]
        if self.property_price_vnd and self.down_payment_vnd:
            lines += [
                f"  Giá trị BĐS:        {fmt(self.property_price_vnd)}",
                f"  Trả trước:          {fmt(self.down_payment_vnd)} ({self.down_payment_pct:.0f}%)",
            ]
        return "\n".join(lines)


@dataclass
class AffordabilityResult:
    """Result of affordability check given income."""
    monthly_income_vnd: float
    dti_ratio: float               # Debt-to-income ratio used (default 0.4)
    max_monthly_payment_vnd: float
    max_loan_vnd: float            # Max borrowable at given rate+term
    annual_rate_pct: float
    term_months: int

    def summary_text(self) -> str:
        def fmt(v):
            b = v / 1_000_000_000
            m = v / 1_000_000
            return f"{b:.2f} tỷ" if b >= 1 else f"{m:.0f} triệu"

        return (
            "=== NĂNG LỰC TÀI CHÍNH (chính xác) ===\n"
            f"  Thu nhập hàng tháng: {fmt(self.monthly_income_vnd)}\n"
            f"  Tỷ lệ DTI an toàn:   {self.dti_ratio*100:.0f}% thu nhập\n"
            f"  Trả tối đa/tháng:    {fmt(self.max_monthly_payment_vnd)}\n"
            f"  Vay tối đa:          {fmt(self.max_loan_vnd)}\n"
            f"  (tại lãi suất {self.annual_rate_pct:.1f}%/năm, {self.term_months//12} năm)"
        )


# ---------------------------------------------------------------------------
# Core calculator
# ---------------------------------------------------------------------------

# Default assumptions (Vietnamese bank market, 2024-2025)
DEFAULT_ANNUAL_RATE_PCT = 9.0       # ~9% average fixed rate
DEFAULT_TERM_YEARS      = 20        # 20 năm
DEFAULT_DTI_RATIO       = 0.40      # 40% thu nhập hàng tháng
DEFAULT_DOWN_PAYMENT_PCT = 0.30     # 30% trả trước


class FinanceCalculator:
    """
    Exact financial calculations for real estate loans.
    All math is deterministic — no LLM involved.
    """

    @staticmethod
    def monthly_payment(
        principal_vnd: float,
        annual_rate_pct: float = DEFAULT_ANNUAL_RATE_PCT,
        term_years: int = DEFAULT_TERM_YEARS,
    ) -> LoanPlan:
        """
        Calculate monthly mortgage payment using the standard PMT formula.

        M = P * [r(1+r)^n] / [(1+r)^n - 1]
        """
        r = (annual_rate_pct / 100) / 12  # monthly rate
        n = term_years * 12               # total months

        if r == 0:
            monthly = principal_vnd / n
        else:
            monthly = principal_vnd * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

        total = monthly * n
        interest = total - principal_vnd

        return LoanPlan(
            principal_vnd=principal_vnd,
            annual_rate_pct=annual_rate_pct,
            term_months=n,
            monthly_payment_vnd=monthly,
            total_payment_vnd=total,
            total_interest_vnd=interest,
        )

    @staticmethod
    def loan_from_property_price(
        property_price_vnd: float,
        down_payment_pct: float = DEFAULT_DOWN_PAYMENT_PCT,
        annual_rate_pct: float = DEFAULT_ANNUAL_RATE_PCT,
        term_years: int = DEFAULT_TERM_YEARS,
    ) -> LoanPlan:
        """Calculate loan plan given a property price and down payment %."""
        down = property_price_vnd * down_payment_pct
        principal = property_price_vnd - down
        plan = FinanceCalculator.monthly_payment(principal, annual_rate_pct, term_years)
        plan.property_price_vnd = property_price_vnd
        plan.down_payment_vnd = down
        plan.down_payment_pct = down_payment_pct * 100
        return plan

    @staticmethod
    def max_affordable_loan(
        monthly_income_vnd: float,
        annual_rate_pct: float = DEFAULT_ANNUAL_RATE_PCT,
        term_years: int = DEFAULT_TERM_YEARS,
        dti_ratio: float = DEFAULT_DTI_RATIO,
    ) -> AffordabilityResult:
        """
        Calculate maximum loan given monthly income and DTI ratio.

        P = M * [(1+r)^n - 1] / [r(1+r)^n]
        """
        max_monthly = monthly_income_vnd * dti_ratio
        r = (annual_rate_pct / 100) / 12
        n = term_years * 12

        if r == 0:
            max_loan = max_monthly * n
        else:
            max_loan = max_monthly * ((1 + r) ** n - 1) / (r * (1 + r) ** n)

        return AffordabilityResult(
            monthly_income_vnd=monthly_income_vnd,
            dti_ratio=dti_ratio,
            max_monthly_payment_vnd=max_monthly,
            max_loan_vnd=max_loan,
            annual_rate_pct=annual_rate_pct,
            term_months=n,
        )

    @staticmethod
    def multi_scenario(
        principal_vnd: float,
        rates: list[float] = None,
        term_years: int = DEFAULT_TERM_YEARS,
    ) -> list[LoanPlan]:
        """
        Compute loan plans for multiple interest rate scenarios.
        Useful for showing bank rate comparisons.
        """
        if rates is None:
            rates = [7.0, 8.5, 9.0, 10.5, 12.0]
        return [
            FinanceCalculator.monthly_payment(principal_vnd, r, term_years)
            for r in rates
        ]
