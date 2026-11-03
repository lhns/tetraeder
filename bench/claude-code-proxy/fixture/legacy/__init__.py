"""Legacy usage-based billing engine.

Still runs the metered plans. Flow: usage events -> metering.aggregate -> pricing.engine -> billing.statements.
Amounts are Decimal dollars, quantized to cents only when a statement line is built.
"""

__version__ = "2.7.3"
