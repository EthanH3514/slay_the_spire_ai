#pragma once
// Public evidence only: an enemy may announce its next move and then die to
// start-turn damage before the next observed decision. Death does not erase
// that observed intent transition. An unchanged dead intent is NOT a new move.
namespace spire_session {
inline constexpr const char* enemyDeathHistoryVersion="public-enemy-death-history-v1";
inline bool publicEnemyMoveAdvanced(bool wasGone,bool nowGone,bool turnChanged,bool intentChanged) {
    return !wasGone && (intentChanged || (!nowGone && turnChanged));
}
}
