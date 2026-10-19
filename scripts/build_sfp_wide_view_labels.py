#!/usr/bin/env python3
"""Score saved SFP wide views against simulator geometry, after blind RGB review."""
import build_sfp_side_view_labels as shared

shared.STAGES = ('baseline_view', 'wide_view_hold')
shared.CASES = (('card1_port0_v2', 1, 0, 'card1_port0'),
                ('card2_port0', 2, 0, 'card2_port0'),
                ('card3_port0', 3, 0, 'card3_port0'),
                ('card4_port1', 4, 1, 'card4_port1'))
shared.EVENT_FILE = 'wide_view_events.json'
shared.SCHEMA = 'sfp_wide_view_event_join/v1'

if __name__ == '__main__':
    shared.main()
