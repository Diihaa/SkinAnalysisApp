# lists containing skin type and skin data

def skin_types():

    types = [
        'Oily skin',
        'Oily',
        'Dry skin',
        'Dry',
        'Sensitive skin',
        'Sensitive',
        'Combination skin',
        'Combination',
        'Normal skin',
        'Normal'
    ]

    return types

def skin_disorders():

    disorders = [
        'Acne',
        'Actinic Keratosis',
        'Actinic keratoses',
        'Adult-onset dermatomyositis',
        'Ageing skin',
        'Allergic contact dermatitis',
        'Angioedema',
        'Angiofibroma',
        'Atopic dermatitis',
        'Basal Cell Carcinoma',
        'Blepharitis',
        'Blushing',
        'Cutaneous lupus erythematosus',
        'Demodicosis',
        'Dermatitis',
        'Discoid lupus erythematosus',
        'Dry skin',
        'Eczema',
        'Eczemaa',
        'Erythema dyschromicum perstans',
        'Flushing',
        'Granuloma faciale',
        'Hair follicle tumours',
        'Jessner lymphocytic infiltrate',
        'Keratosis pilaris atrophicans faciei',
        'Lichen planus pigmentosus',
        'Melasma',
        'Neonatal lupus erythematosus',
        'Perioral/periorificial dermatitis',
        'Photosensitivity',
        'Pityriasis alba',
        'Poikiloderma of Civatte',
        'Pseudofolliculitis barbae',
        'Psoriasis',
        'Pyoderma faciale',
        'Rosacea',
        'Sarcoidosis',
        'Sebaceous hyperplasia',
        'Seborrhoea',
        'Seborrhoeic dermatitis',
        'Sensitive skin',
        'Shaving rash',
        'Solar keratoses',
        'Solar/senile comedones',
        'Steroid acne',
        'Steroid rosacea',
        'Systemic lupus erythematosus',
        'Trigeminal trophic syndrome',
        'Vitiligo'
    ]

    return disorders

def recommendations(skin_type):

    recommendations_data = {
        "Oily Skin": {
            "Cleansing": "Use a gentle foaming or gel-based cleanser twice a day to remove excess oil without stripping moisture. Look for cleansers labeled 'oil-free' or 'non-comedogenic' to avoid clogging pores.",
            "Exfoliation": "Exfoliate 2-3 times a week with a gentle chemical exfoliant containing salicylic acid or glycolic acid to unclog pores and prevent breakouts.",
            "Moisturizing": "Use a lightweight, oil-free moisturizer to maintain hydration without adding extra shine. Look for ingredients like hyaluronic acid or glycerin.",
            "Lifestyle": "Avoid harsh scrubs, hot showers, and excessive sun exposure. Manage stress, and maintain a healthy diet rich in fruits and vegetables.",
        },
        "Oily": {
            "Cleansing": "Use a gentle foaming or gel-based cleanser twice a day to remove excess oil without stripping moisture. Look for cleansers labeled 'oil-free' or 'non-comedogenic' to avoid clogging pores.",
            "Exfoliation": "Exfoliate 2-3 times a week with a gentle chemical exfoliant containing salicylic acid or glycolic acid to unclog pores and prevent breakouts.",
            "Moisturizing": "Use a lightweight, oil-free moisturizer to maintain hydration without adding extra shine. Look for ingredients like hyaluronic acid or glycerin.",
            "Lifestyle": "Avoid harsh scrubs, hot showers, and excessive sun exposure. Manage stress, and maintain a healthy diet rich in fruits and vegetables.",
        },
        "Dry Skin": {
            "Cleansing": "Use a hydrating, creamy cleanser once a day to remove dirt and impurities without over-drying. Avoid harsh soaps and cleansers with alcohol.",
            "Exfoliation": "Exfoliate 1-2 times a week with a gentle physical or chemical exfoliant to remove dead skin cells. Look for ingredients like lactic acid.",
            "Moisturizing": "Apply a rich, hydrating moisturizer twice a day, with additional application throughout the day if needed. Look for ingredients like ceramides, hyaluronic acid, and shea butter.",
            "Lifestyle": "Take lukewarm showers and baths, use a humidifier, and drink plenty of water to stay hydrated. Limit caffeine and alcohol intake.",
        },
        "Dry": {
            "Cleansing": "Use a hydrating, creamy cleanser once a day to remove dirt and impurities without over-drying. Avoid harsh soaps and cleansers with alcohol.",
            "Exfoliation": "Exfoliate 1-2 times a week with a gentle physical or chemical exfoliant to remove dead skin cells. Look for ingredients like lactic acid.",
            "Moisturizing": "Apply a rich, hydrating moisturizer twice a day, with additional application throughout the day if needed. Look for ingredients like ceramides, hyaluronic acid, and shea butter.",
            "Lifestyle": "Take lukewarm showers and baths, use a humidifier, and drink plenty of water to stay hydrated. Limit caffeine and alcohol intake.",
        },
        "Sensitive Skin": {
            "Cleansing": "Use a gentle, fragrance-free cleanser twice a day. Avoid harsh chemicals, scrubs, and products with alcohol or artificial fragrances.",
            "Exfoliation": "Exfoliate very gently, no more than once a week. Patch test new products before applying them to your entire face.",
            "Moisturizing": "Use a fragrance-free, hypoallergenic moisturizer to soothe and protect your skin. Look for ingredients like colloidal oatmeal, ceramides, and aloe vera.",
            "Lifestyle": "Identify and avoid triggers that irritate your skin, such as certain foods, harsh weather conditions, or stress. Wear sunscreen daily, even on cloudy days.",
        },
        "Sensitive": {
            "Cleansing": "Use a gentle, fragrance-free cleanser twice a day. Avoid harsh chemicals, scrubs, and products with alcohol or artificial fragrances.",
            "Exfoliation": "Exfoliate very gently, no more than once a week. Patch test new products before applying them to your entire face.",
            "Moisturizing": "Use a fragrance-free, hypoallergenic moisturizer to soothe and protect your skin. Look for ingredients like colloidal oatmeal, ceramides, and aloe vera.",
            "Lifestyle": "Identify and avoid triggers that irritate your skin, such as certain foods, harsh weather conditions, or stress. Wear sunscreen daily, even on cloudy days.",
        },
        "Combination Skin": {
            "Cleansing": "Use a gentle cleanser that won't over-dry or over-moisturize. Micellar water can be a good option. Cleanse twice a day.",
            "Exfoliation": "Exfoliate 2-3 times a week, but use different methods for oily and dry areas. A gentle chemical exfoliant for the T-zone and a physical exfoliant for the cheeks might be suitable.",
            "Moisturizing": "Use a lightweight, oil-free moisturizer for the T-zone and a richer moisturizer for the drier cheeks. Look for products labeled 'combination skin'.",
            "Lifestyle": "Focus on balancing oil production. Manage stress, eat a healthy diet, and avoid harsh products or excessive sun exposure.",
        },
        "Combination": {
            "Cleansing": "Use a gentle cleanser that won't over-dry or over-moisturize. Micellar water can be a good option. Cleanse twice a day.",
            "Exfoliation": "Exfoliate 2-3 times a week, but use different methods for oily and dry areas. A gentle chemical exfoliant for the T-zone and a physical exfoliant for the cheeks might be suitable.",
            "Moisturizing": "Use a lightweight, oil-free moisturizer for the T-zone and a richer moisturizer for the drier cheeks. Look for products labeled 'combination skin'.",
            "Lifestyle": "Focus on balancing oil production. Manage stress, eat a healthy diet, and avoid harsh products or excessive sun exposure.",
        },
        "Normal Skin": {
            "Cleansing": "Use a gentle cleanser twice a day to remove dirt and makeup. Cream or gel cleansers can both work well for normal skin.",
            "Exfoliation": "Exfoliate 1-2 times a week with a gentle exfoliant to remove dead skin cells and promote radiance.",
            "Moisturizing": "Use a lightweight, oil-free moisturizer daily to maintain hydration. Look for ingredients like hyaluronic acid or glycerin.",
            "Lifestyle": "Maintain a healthy diet and lifestyle habits for overall skin health. Don't forget sun protection with a daily SPF 30+ sunscreen.",
        },
        "Normal": {
            "Cleansing": "Use a gentle cleanser twice a day to remove dirt and makeup. Cream or gel cleansers can both work well for normal skin.",
            "Exfoliation": "Exfoliate 1-2 times a week with a gentle exfoliant to remove dead skin cells and promote radiance.",
            "Moisturizing": "Use a lightweight, oil-free moisturizer daily to maintain hydration. Look for ingredients like hyaluronic acid or glycerin.",
            "Lifestyle": "Maintain a healthy diet and lifestyle habits for overall skin health. Don't forget sun protection with a daily SPF 30+ sunscreen.",
        },
    }

    recommended_routine = recommendations_data[skin_type]

    return recommended_routine