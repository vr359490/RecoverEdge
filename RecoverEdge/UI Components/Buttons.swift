//
//  Buttons.swift
//  RecoverEdge
//
//  Created by Victor Ruan on 6/26/25.
//

import SwiftUI
import WebKit

// MARK: - UI Components
struct TimeButton: View {
    let time: Int
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
                VStack(spacing: 0) {
                    Text("\(time)")
                        .font(.system(size: 28, weight: .bold))
                    Text("MIN")
                        .font(.system(size: 18, weight: .bold))
                }
                .multilineTextAlignment(.center)
                .padding(.horizontal, 39)
                .padding(.vertical, 30)
                .background(Group {
                    if isSelected {
                        LinearGradient(
                            colors: [Color.brandTeal, Color.brandTealDark,Color(r:30, g:80, b:80), Color.black],
                            startPoint: .bottom,
                            endPoint: .top
                        )
                    } else {
                        LinearGradient(
                            colors: [Color(r:98, g:252, b:236),Color.brandTeal,Color(r:80, g:190, b:190), Color(r:50, g:123, b:127)],
                            startPoint: .bottom,
                            endPoint: .top
                        )
                    }
                })
                .foregroundColor(isSelected ? .white : .primary)
                .cornerRadius(20)
        }
    }
}

struct LocationButton: View {
    let location: Location
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: iconForLocation(location))
                    .font(.title2)
                Text(location.rawValue)
                    .font(.caption)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(
                Group {
                if isSelected {
                    LinearGradient(
                        colors: [Color.brandTeal2, Color.brandTealDark,Color(r:30, g:80, b:80), Color.black],
                        startPoint: .bottom,
                        endPoint: .top
                    )
                } else {
                    LinearGradient(
                        colors: [Color(r:98, g:252, b:236),Color.brandTeal,Color(r:80, g:190, b:190), Color(r:50, g:123, b:127)],
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
        case .outdoors: return "tree"
        case .none: return ""
        }
    }
}

struct EquipmentButton: View {
    let equipment: Equipment
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(equipment.name)
                .font(.caption)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    Group {
                    if isSelected {
                        LinearGradient(
                            colors: [Color.brandTealDark,Color(r:30, g:80, b:80), Color.black],
                            startPoint: .bottom,
                            endPoint: .top
                        )
                    } else {
                        Color.gray.opacity(0.2)
                    }
                })
                .foregroundColor(isSelected ? .white : .primary)
                .cornerRadius(8)
        }
    }
}
