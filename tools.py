import json

def get_customer_orders(customer_id: str) -> str:
    """Retrieves customer account details and order history."""
    database = {
        "CUST101": {
            "name": "Alice Smith", 
            "status": "VIP", 
            "email": "alice@example.com",
            "orders": [{"id": "ORD-99", "item": "Wireless Headphones", "amount": 120.0, "status": "Delivered"}]
        },
        "CUST102": {
            "name": "Bob Jones", 
            "status": "Regular", 
            "email": "bob@example.com",
            "orders": [{"id": "ORD-100", "item": "Mechanical Keyboard", "amount": 85.0, "status": "In Transit"}]
        }
    }
    data = database.get(customer_id.upper())
    if data:
        return json.dumps(data)
    return json.dumps({"error": f"Customer ID {customer_id} not found."})

def calculate_refund(order_amount: float, discount_percent: float = 0.0) -> str:
    """Calculates refund amount after applying optional discount/penalty deduction."""
    refund = order_amount * (1 - (discount_percent / 100.0))
    return json.dumps({"calculated_refund": round(refund, 2)})

def send_support_email(recipient_email: str, subject: str, message_body: str) -> str:
    """Simulates sending an email to a customer."""
    return json.dumps({
        "status": "success",
        "action": "email_sent",
        "to": recipient_email,
        "subject": subject
    })

def get_all_customers_spending() -> str:
    """Calculates total spend across all database customers."""
    database = {
        "CUST101": {"name": "Alice Smith", "orders": [{"amount": 120.0}]},
        "CUST102": {"name": "Bob Jones", "orders": [{"amount": 85.0}]}
    }
    total_spent = sum(sum(o["amount"] for o in d["orders"]) for d in database.values())
    return json.dumps({"combined_total_spent": total_spent})

AVAILABLE_TOOLS = {
    "get_customer_orders": get_customer_orders,
    "calculate_refund": calculate_refund,
    "send_support_email": send_support_email,
    "get_all_customers_spending": get_all_customers_spending
}