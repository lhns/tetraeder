"""Exception hierarchy for the billing engine."""


class BillingError(Exception):
    """Base class; callers that only want to log can catch this."""


class UnknownCustomer(BillingError):
    def __init__(self, customer_id: str):
        super().__init__(f"unknown customer: {customer_id}")
        self.customer_id = customer_id


class UnknownPlan(BillingError):
    def __init__(self, plan_id: str):
        super().__init__(f"unknown plan: {plan_id}")
        self.plan_id = plan_id


class InvalidUsage(BillingError):
    """A usage event or aggregate is malformed (negative units, unknown meter, bad timestamp)."""


class InvalidPeriod(BillingError):
    """A billing period is empty or reversed."""


class PricingError(BillingError):
    """A plan's tier table is inconsistent (gaps, unordered limits, no open-ended last tier)."""


class PromoError(BillingError):
    """A promo code cannot be applied."""


class ExportError(BillingError):
    """A value does not fit the export format."""
