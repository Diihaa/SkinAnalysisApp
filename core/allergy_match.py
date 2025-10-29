# allergy_match.py

def allergy_match(user_allergies, product_ingredients):
    """
    Compare user allergies with product ingredients.
    Args:
        user_allergies (list of str): e.g. ["nuts", "milk"]
        product_ingredients (list of str): e.g. ["water", "sugar", "nuts"]
    Returns:
        bool: True if product is safe, False if it contains allergens
    """
    for allergy in user_allergies:
        if allergy.strip().lower() in [ing.strip().lower() for ing in product_ingredients]:
            return False   # unsafe product (contains allergen)
    return True            # safe product
