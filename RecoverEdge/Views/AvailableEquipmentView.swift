//
//  AvailableEquipmentView.swift
//  RecoverEdge
//
//  Created by Victor Ruan on 7/3/25.
//

import SwiftUI

// MARK: - Equipment Preferences Manager (Updated for AvailableEquipmentView.swift)
class EquipmentPreferencesManager: ObservableObject {
    static let shared = EquipmentPreferencesManager()
    
    @Published var savedEquipment: [Location: Set<String>] = [:]
    
    private let userDefaults = UserDefaults.standard
    private let equipmentKey = "SavedEquipmentPreferences"
    
    private init() {
        loadSavedPreferences()
        setupSmartDefaults()
    }
    
    // MARK: - Updated Smart Defaults with New Equipment
    private func setupSmartDefaults() {
        // Only set defaults if no preferences exist yet
        for location in Location.allCases where location != .none {
            if savedEquipment[location] == nil {
                savedEquipment[location] = getSmartDefaults(for: location)
            }
        }
    }
    
    func getSmartDefaults(for location: Location) -> Set<String> {
        switch location {
        case .gym:
            // Include both basic and advanced equipment for comprehensive gym experience
            return Set([
                // Basic recovery equipment
                "Foam Roller",
                "Yoga Mat",
                "Resistance Bands",
                "Sauna",
                // Advanced equipment for premium gyms
                "Normatec",
                "Vibration Plate",
                "Hypervolt Gun"
            ])
        case .hotel:
            // Focus on available amenities and portable items
            return Set([
                "Wall",
                "Towel",
                "Bathtub",
                "Sauna", // Many hotels have spa facilities
                "Hot Pad" // Travel-friendly recovery tool
            ])
        case .home:
            // Personal equipment that people commonly own
            return Set([
                "Foam Roller",
                "Yoga Mat",
                "Towel",
                "Wall",
                "Resistance Bands",
                "Hot Pad" // Affordable and effective home recovery tool
            ])
        case .court:
            // Basic items available at most courts plus bodyweight options
            return Set([
                "Ground",
                "Water Bottle",
                "Towel",
                "Court Wall"
            ])
        case .none:
            return Set()
        }
    }
    
    // MARK: - Save/Load Methods (unchanged)
    func saveEquipment(for location: Location, equipment: Set<String>) {
        savedEquipment[location] = equipment
        saveToUserDefaults()
    }
    
    func getEquipment(for location: Location) -> Set<String> {
        return savedEquipment[location] ?? getSmartDefaults(for: location)
    }
    
    private func saveToUserDefaults() {
        // Convert Location keys to String keys for encoding
        let encodableDict = Dictionary(uniqueKeysWithValues:
            savedEquipment.map { (location, equipment) in
                (location.rawValue, Array(equipment))
            }
        )
        
        if let data = try? JSONEncoder().encode(encodableDict) {
            userDefaults.set(data, forKey: equipmentKey)
        }
    }
    
    private func loadSavedPreferences() {
        guard let data = userDefaults.data(forKey: equipmentKey),
              let decoded = try? JSONDecoder().decode([String: [String]].self, from: data) else {
            return
        }
        
        for (locationString, equipmentArray) in decoded {
            if let location = Location(rawValue: locationString) {
                savedEquipment[location] = Set(equipmentArray)
            }
        }
    }
    
    // MARK: - Helper Methods (unchanged)
    func hasCustomizedEquipment(for location: Location) -> Bool {
        let current = getEquipment(for: location)
        let defaults = getSmartDefaults(for: location)
        return current != defaults
    }
    
    func resetToDefaults(for location: Location) {
        savedEquipment[location] = getSmartDefaults(for: location)
        saveToUserDefaults()
    }
}

// MARK: - Enhanced Available Equipment View
struct AvailableEquipmentView: View {
    @StateObject private var preferencesManager = EquipmentPreferencesManager.shared
    @Environment(\.presentationMode) var presentationMode
    
    @State private var selectedLocation: Location = .gym // Start with gym instead of .none
    @State private var tempSelectedEquipment: Set<String> = []
    @State private var showingResetAlert = false
    
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Header with description
                VStack(spacing: 8) {
                    Text("Configure Your Equipment")
                        .font(.title2)
                        .fontWeight(.semibold)
                    
                    Text("Set up what equipment you have access to at each location. This will be remembered for future recovery sessions.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                .padding(.top, 10)
                
                // Location Selection
                LocationSelectionSection(
                    selectedLocation: $selectedLocation,
                    onLocationChanged: { newLocation in
                        saveCurrentSelection()
                        selectedLocation = newLocation
                        loadEquipmentForLocation()
                    }
                )
                
                // Equipment Selection
                EquipmentConfigurationSection(
                    location: selectedLocation,
                    selectedEquipment: $tempSelectedEquipment,
                    onResetToDefaults: {
                        showingResetAlert = true
                    }
                )
                
                Spacer(minLength: 100)
            }
        }
        .navigationTitle("Available Equipment")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("Save") {
                    saveCurrentSelection()
                    presentationMode.wrappedValue.dismiss()
                }
                .fontWeight(.semibold)
            }
        }
        .onAppear {
            loadEquipmentForLocation()
        }
        .onDisappear {
            saveCurrentSelection()
        }
        .alert("Reset to Defaults", isPresented: $showingResetAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Reset", role: .destructive) {
                preferencesManager.resetToDefaults(for: selectedLocation)
                loadEquipmentForLocation()
            }
        } message: {
            Text("This will reset your equipment selection for \(selectedLocation.rawValue) to the recommended defaults.")
        }
    }
    
    private func loadEquipmentForLocation() {
        tempSelectedEquipment = preferencesManager.getEquipment(for: selectedLocation)
    }
    
    private func saveCurrentSelection() {
        preferencesManager.saveEquipment(for: selectedLocation, equipment: tempSelectedEquipment)
    }
}

// MARK: - Location Selection Section (for settings)
struct LocationSelectionSection: View {
    @Binding var selectedLocation: Location
    let onLocationChanged: (Location) -> Void
    @StateObject private var preferencesManager = EquipmentPreferencesManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Location")
                    .font(.headline)
                Text("Choose a location to configure")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal)
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(Location.allCases.filter { $0 != .none }, id: \.self) { location in
                        LocationButtonWithIndicator(
                            location: location,
                            isSelected: selectedLocation == location,
                            hasCustomEquipment: preferencesManager.hasCustomizedEquipment(for: location),
                            action: {
                                onLocationChanged(location)
                            }
                        )
                    }
                }
                .padding(.horizontal)
            }
        }
    }
}

// MARK: - Location Button with Customization Indicator
struct LocationButtonWithIndicator: View {
    let location: Location
    let isSelected: Bool
    let hasCustomEquipment: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                ZStack {
                    Image(systemName: iconForLocation(location))
                        .font(.title2)
                    
                    
                    // Not so helpful
//                    // Customization indicator
//                    if hasCustomEquipment {
//                        VStack {
//                            HStack {
//                                Spacer()
//                                Circle()
//                                    .fill(Color.orange)
//                                    .frame(width: 8, height: 8)
//                            }
//                            Spacer()
//                        }
//                        .frame(width: 30, height: 30)
//                    }
                }
                
                Text(location.rawValue)
                    .font(.caption)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(
                Group {
                    if isSelected {
                        LinearGradient(
                            colors: [Color.brandTeal2, Color.brandTealDark, Color(r:30, g:80, b:80), Color.black],
                            startPoint: .bottom,
                            endPoint: .top
                        )
                    } else {
                        LinearGradient(
                            colors: [Color(r:98, g:252, b:236), Color.brandTeal, Color(r:80, g:190, b:190), Color(r:50, g:123, b:127)],
                            startPoint: .bottom,
                            endPoint: .top
                        )
                    }
                }
            )
            .foregroundColor(isSelected ? .white : .primary)
            .cornerRadius(12)
        }
    }
    
    private func iconForLocation(_ location: Location) -> String {
        switch location {
        case .gym: return "dumbbell"
        case .hotel: return "bed.double"
        case .home: return "house"
        case .court: return "sportscourt"
        case .none: return ""
        }
    }
}

// MARK: - Equipment Configuration Section
struct EquipmentConfigurationSection: View {
    let location: Location
    @Binding var selectedEquipment: Set<String>
    let onResetToDefaults: () -> Void
    @StateObject private var preferencesManager = EquipmentPreferencesManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Section header with controls
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Equipment at \(location.rawValue)")
                        .font(.headline)
                    Text("\(selectedEquipment.count) of \(location.availableEquipment.count) selected")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Menu {
                    Button("Select All") {
                        selectedEquipment = Set(location.availableEquipment.map { $0.name })
                    }
                    
                    Button("Clear All") {
                        selectedEquipment.removeAll()
                    }
                    
                    Divider()
                    
                    Button("Reset to Defaults") {
                        onResetToDefaults()
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.title3)
                        .foregroundColor(.brandTeal)
                }
            }
            .padding(.horizontal)
            
            // Equipment grid
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                ForEach(location.availableEquipment, id: \.id) { equipment in
                    EquipmentButton(
                        equipment: equipment,
                        isSelected: selectedEquipment.contains(equipment.name),
                        action: {
                            if selectedEquipment.contains(equipment.name) {
                                selectedEquipment.remove(equipment.name)
                            } else {
                                selectedEquipment.insert(equipment.name)
                            }
                        }
                    )
                }
            }
            .padding(.horizontal)
            
            // Smart suggestions
            if !suggestedEquipment.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Suggested for most users:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.horizontal)
                    
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(suggestedEquipment, id: \.self) { equipmentName in
                                Button(action: {
                                    selectedEquipment.insert(equipmentName)
                                }) {
                                    Text(equipmentName)
                                        .font(.caption)
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 4)
                                        .background(Color.green.opacity(0.1))
                                        .foregroundColor(.green)
                                        .cornerRadius(6)
                                }
                                .disabled(selectedEquipment.contains(equipmentName))
                            }
                        }
                        .padding(.horizontal)
                    }
                }
            }
        }
        .animation(.easeInOut(duration: 0.2), value: selectedEquipment)
    }
    
    private var suggestedEquipment: [String] {
        let defaults = preferencesManager.getSmartDefaults(for: location)
        return Array(defaults.subtracting(selectedEquipment))
    }
}
