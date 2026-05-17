"""
Travel knowledge base documents seeded into ChromaDB.
These provide RAG context to the planning agents for richer, more accurate itineraries.
"""

TRAVEL_DOCUMENTS = [
    {
        "id": "budget-planning-001",
        "content": """Budget travel planning fundamentals:
        - Rule of thumb: accommodation typically takes 30-40% of total budget
        - Food budget: street food averages $5-15/day, mid-range restaurants $20-40/day
        - Always keep 10-15% emergency buffer in travel budget
        - Book accommodation 2-4 weeks in advance for best rates
        - City passes often provide 20-40% savings on attractions
        - Free walking tours available in most major cities (tip-based)
        - Public transport is usually 5-10x cheaper than taxis/rideshares
        - Travel on weekdays for cheaper flights and less crowded attractions""",
        "metadata": {"category": "budget", "type": "general"},
    },
    {
        "id": "safety-travel-001",
        "content": """Travel safety best practices:
        - Always register with your country's embassy when visiting high-risk areas
        - Keep digital copies of passport, visa, insurance in cloud storage
        - Use ATMs inside banks rather than street ATMs to avoid skimming
        - Carry split cash: some in wallet, some in hotel safe
        - Share your itinerary with someone at home
        - Purchase comprehensive travel insurance covering medical evacuation
        - Learn basic local phrases: help, emergency, hospital, police
        - Keep emergency contact numbers saved offline
        - Avoid displaying expensive jewelry or electronics in crowded areas""",
        "metadata": {"category": "safety", "type": "general"},
    },
    {
        "id": "packing-essentials-001",
        "content": """Smart packing strategies:
        - Pack clothes you can layer for temperature changes
        - Bring universal power adapter and portable charger
        - Microfiber towel saves luggage space
        - Compression bags reduce clothing volume by 60%
        - Always pack medications in carry-on, never checked luggage
        - Reusable water bottle with filter saves money and is eco-friendly
        - Download offline maps before departure (Google Maps offline)
        - Photocopy important documents and store separately from originals
        - Pack one formal outfit for unexpected events or nicer restaurants""",
        "metadata": {"category": "packing", "type": "general"},
    },
    {
        "id": "transport-global-001",
        "content": """Global transportation tips:
        - Japan: IC cards (Suica/Pasmo) work on almost all public transport
        - Europe: Eurail passes worthwhile for 5+ countries, point-to-point tickets otherwise
        - Southeast Asia: Grab app is the safe rideshare across the region
        - USA: Uber/Lyft in cities, car rental essential outside major metros
        - Middle East: Careem is the dominant rideshare app
        - Always screenshot/download transport QR codes before trips
        - Overnight trains or buses save both money and accommodation costs
        - Airport express trains are almost always faster and cheaper than taxis""",
        "metadata": {"category": "transport", "type": "general"},
    },
    {
        "id": "food-culture-001",
        "content": """Food and dining travel tips:
        - Eat where locals eat: look for full restaurants at lunchtime
        - Markets and food halls offer authentic food at lower prices
        - Learn dietary restriction phrases in the local language
        - Tipping customs vary: mandatory in USA, offensive in Japan, optional in Europe
        - Street food is often safer than it looks — high turnover means fresh food
        - Lunch menus at nice restaurants often cost 30-50% less than dinner
        - Food tours on day 1 or 2 help you discover best local spots
        - Avoid tourist-trap restaurants within 200m of major attractions""",
        "metadata": {"category": "food", "type": "general"},
    },
    {
        "id": "cultural-etiquette-001",
        "content": """Cultural etiquette essentials:
        - Research dress codes before visiting religious sites (shoulders/knees covered)
        - In Japan: no eating while walking, shoes off indoors, quiet on public transport
        - In Middle East: public displays of affection are restricted
        - In India: remove shoes at temples and some homes
        - In Thailand: never touch someone's head, point feet away from sacred objects
        - In France: greet shopkeepers when entering, say goodbye when leaving
        - Photography: always ask permission before photographing people
        - Bargaining: expected in markets in Asia/Africa/Middle East, not in Europe""",
        "metadata": {"category": "culture", "type": "general"},
    },
    {
        "id": "asia-travel-001",
        "content": """Asia travel insights:
        - Best time: Oct-April for Southeast Asia (dry season), April-May for Japan (cherry blossoms)
        - Visa on arrival: Thailand (60 days), Indonesia (30 days), Sri Lanka (30 days)
        - Japan is expensive but safe; budget $150-200/day including accommodation
        - Vietnam and Cambodia offer exceptional value: $40-60/day comfortable budget
        - Singapore: expensive but efficient — great transit hub, 3-day stopover ideal
        - India: best experienced with a local guide for first-timers
        - China: requires VPN, WeChat Pay, cash; plan 2-3 weeks for highlights""",
        "metadata": {"category": "destination", "region": "asia"},
    },
    {
        "id": "europe-travel-001",
        "content": """Europe travel insights:
        - Schengen visa allows 90 days across 26 countries
        - Best value countries: Portugal, Poland, Czech Republic, Hungary, Greece
        - Summer (Jun-Aug): crowded and expensive; shoulder season (Apr-May, Sep-Oct) ideal
        - Budget airlines: Ryanair, EasyJet connect cities cheaply
        - Hostels with private rooms offer great value and social atmosphere
        - Museum passes: Paris Museum Pass, Amsterdam City Card save significant money
        - Rail travel best for distances under 600km; fly for longer routes
        - Book popular sites (Eiffel Tower, Sagrada Familia) weeks in advance online""",
        "metadata": {"category": "destination", "region": "europe"},
    },
    {
        "id": "americas-travel-001",
        "content": """Americas travel insights:
        - USA: vast distances mean internal flights often necessary; rent car outside major cities
        - Mexico: safe in tourist areas; Oaxaca, Merida, San Miguel de Allende highly recommended
        - Colombia: Cartagena, Medellin transformed — vibrant food and nightlife scenes
        - Peru: Machu Picchu requires advance booking (huaynapicchu tickets sell out months ahead)
        - Brazil: Portuguese language barrier; Visa required for some nationalities
        - Costa Rica: expensive but stunning; ideal for nature/adventure lovers
        - Canada: budget similar to USA but stunning national parks worth the trip""",
        "metadata": {"category": "destination", "region": "americas"},
    },
    {
        "id": "wellness-travel-001",
        "content": """Health and wellness travel tips:
        - Travel insurance with medical coverage is non-negotiable
        - Vaccinations: check CDC or NHS recommendations 6-8 weeks before travel
        - Jet lag: adjust sleep schedule 3 days before departure
        - Stay hydrated on flights: drink 250ml water per hour of flying
        - Walk 10,000+ steps daily while traveling helps with jet lag and mood
        - Many countries have excellent, affordable pharmacies for minor ailments
        - Meditation apps help with travel anxiety and adjusting to new environments""",
        "metadata": {"category": "wellness", "type": "general"},
    },
]
