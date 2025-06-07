const tripDataRaw = localStorage.getItem("tripData");

// Synthetic database with realistic data
const database = {
    facilities: [
        { name: "YellowSquare Rome", type: "Hostel", costPerNight: 25, city: "Rome", rating: 4.5, image: "hostel-rome.png", location: "Near Termini Station" },
        { name: "Generator Paris", type: "Hostel", costPerNight: 30, city: "Paris", rating: 4.3, image: "hostel-paris.png", location: "Near Gare du Nord" },
        { name: "St. Christopher's Barcelona", type: "Hostel", costPerNight: 28, city: "Barcelona", rating: 4.4, image: "hostel-barcelona.png", location: "Near La Rambla" },
        { name: "Hotel Roma", type: "Hotel", costPerNight: 80, city: "Rome", rating: 4.2, image: "hotel-rome.png", location: "City Center" },
        { name: "Hotel Paris", type: "Hotel", costPerNight: 90, city: "Paris", rating: 4.1, image: "hotel-paris.png", location: "Near Eiffel Tower" },
        { name: "Airbnb Rome Center", type: "Apartment", costPerNight: 60, city: "Rome", rating: 4.6, image: "apartment-rome.png", location: "Trastevere" }
    ],
    transports: [
        { name: "Ryanair", type: "Plane", duration: "2h 30m", cost: 50, from: "London", to: "Rome" },
        { name: "Eurostar", type: "Train", duration: "2h 15m", cost: 80, from: "London", to: "Paris" },
        { name: "FlixBus", type: "Bus", duration: "3h", cost: 25, from: "Paris", to: "Rome" },
        { name: "Trenitalia", type: "Train", duration: "1h 30m", cost: 35, from: "Rome", to: "Florence" }
    ],
    activities: [
        { name: "Colosseum Tour", type: "Historical Sites", duration: "3h", cost: 20, city: "Rome", image: "colosseum.png", description: "Skip-the-line tour of the ancient Roman amphitheater" },
        { name: "Vatican Museums", type: "Art & Museums", duration: "4h", cost: 25, city: "Rome", image: "vatican.png", description: "World-famous art collection including the Sistine Chapel" },
        { name: "Eiffel Tower", type: "Historical Sites", duration: "2h", cost: 30, city: "Paris", image: "eiffel.png", description: "Iconic symbol of Paris with panoramic views" },
        { name: "Louvre Museum", type: "Art & Museums", duration: "3h", cost: 15, city: "Paris", image: "louvre.png", description: "Home to the Mona Lisa and other masterpieces" },
        { name: "Trastevere Food Tour", type: "Local Food", duration: "3h", cost: 45, city: "Rome", image: "food-rome.png", description: "Authentic Italian cuisine tasting experience" },
        { name: "Latin Quarter Nightlife", type: "Nightlife", duration: "4h", cost: 35, city: "Paris", image: "nightlife-paris.png", description: "Experience Parisian nightlife in the historic district" }
    ]
};

if (tripDataRaw) {
    const tripData = JSON.parse(tripDataRaw);
    const app = document.getElementById("app");
    
    // Calculate trip duration
    const startDate = new Date(tripData.startDate);
    const endDate = new Date(tripData.endDate);
    const days = Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24));
    
    // Calculate daily budget
    const totalBudget = parseInt(tripData.budget);
    const dailyBudget = totalBudget / days;
    
    // Generate itinerary
    let resultHTML = "";
    let currentDate = new Date(startDate);
    
    // Generate day-by-day itinerary
    for (let i = 0; i < days; i++) {
        const city = tripData.destinations[Math.min(i, tripData.destinations.length - 1)];
        const dateStr = currentDate.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' });
        
        // Find suitable accommodation
        const accommodation = database.facilities.find(f => 
            f.city === city && 
            tripData.preferences.accommodation.includes(f.type) &&
            f.costPerNight <= dailyBudget * 0.4
        ) || database.facilities.find(f => f.city === city);

        // Find suitable activities based on preferences
        const activities = database.activities.filter(a => 
            a.city === city && 
            tripData.preferences.interests.includes(a.type) &&
            a.cost <= dailyBudget * 0.6
        ).slice(0, 3);

        // Calculate total cost for the day
        const dayCost = (accommodation ? accommodation.costPerNight : 0) + 
                       activities.reduce((sum, a) => sum + a.cost, 0);

        resultHTML += `
            <div class="day-card">
                <div class="day-header">
                    <h3>Day ${i + 1} - ${city}</h3>
                    <span class="price-tag">€${dayCost}</span>
                </div>
                <div class="activity">
                    <div class="activity-time">09:00</div>
                    <div class="activity-details">
                        <h4>${activities[0]?.name || 'Free Time'}</h4>
                        <p>${activities[0]?.description || 'Explore the city at your own pace'}</p>
                    </div>
                </div>
                <div class="activity">
                    <div class="activity-time">13:00</div>
                    <div class="activity-details">
                        <h4>${activities[1]?.name || 'Lunch Break'}</h4>
                        <p>${activities[1]?.description || 'Enjoy local cuisine'}</p>
                    </div>
                </div>
                <div class="activity">
                    <div class="activity-time">15:00</div>
                    <div class="activity-details">
                        <h4>${activities[2]?.name || 'Evening Activities'}</h4>
                        <p>${activities[2]?.description || 'Experience local culture'}</p>
                    </div>
                </div>
                <div class="accommodation-info">
                    <h4>Accommodation</h4>
                    <p>${accommodation ? accommodation.name : 'To be arranged'}</p>
                    <p>${accommodation ? accommodation.location : ''}</p>
                </div>
            </div>
        `;

        currentDate.setDate(currentDate.getDate() + 1);
    }

    // Add transportation information
    const transportInfo = database.transports.find(t => 
        t.from === tripData.from && 
        t.to === tripData.destinations[0] &&
        tripData.preferences.transport.includes(t.type)
    );

    if (transportInfo) {
        resultHTML = `
            <div class="transport-info">
                <h3>Transportation Details</h3>
                <p>${transportInfo.name} - ${transportInfo.type}</p>
                <p>Duration: ${transportInfo.duration}</p>
                <p>Cost: €${transportInfo.cost}</p>
            </div>
        ` + resultHTML;
    }

    // Add total cost and student discount
    const totalCost = days * dailyBudget;
    const studentDiscount = totalCost * 0.2;
    const finalCost = totalCost - studentDiscount;

    resultHTML = `
        <div class="trip-summary">
            <div class="trip-header">
                <div>
                    <h2>${tripData.from} → ${tripData.destinations.join(' → ')}</h2>
                    <div class="trip-dates">${startDate.toLocaleDateString()} - ${endDate.toLocaleDateString()}</div>
                </div>
                <div class="student-discount">
                    <img src="student-icon.png" alt="Student" style="width: 20px; vertical-align: middle;">
                    Student Discount Applied
                </div>
            </div>
            <div class="trip-stats">
                <div class="stat-card">
                    <h3>${days} Days</h3>
                    <p>Duration</p>
                </div>
                <div class="stat-card">
                    <h3>${tripData.destinations.length} Cities</h3>
                    <p>Destinations</p>
                </div>
                <div class="stat-card">
                    <h3>€${totalCost}</h3>
                    <p>Total Budget</p>
                </div>
                <div class="stat-card">
                    <h3>20% Off</h3>
                    <p>Student Discount</p>
                </div>
            </div>
        </div>
    ` + resultHTML;

    app.innerHTML = resultHTML;
} else {
    console.log("No trip data found");
    window.location.href = 'Prart1.html';
}
