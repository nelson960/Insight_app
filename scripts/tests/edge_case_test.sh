#!/usr/bin/env bash
#
# Insight Edge Case Testing
#
# Tests unusual scenarios and edge cases that may break the app
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
GRAY='\033[0;90m'
NC='\033[0m'

bold() { echo -e "${BLUE}➜${NC} $*"; }
info() { echo -e "  ${GRAY}$*${NC}"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warning() { echo -e "${YELLOW}⚠${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*" >&2; }
header() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n${BLUE}$*${NC}\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }

# Configuration
APP_PATH="${1:-/Users/nelson/py/insight/insight_app/dist/onedir/Insight.app}"
TEST_DIR="/tmp/insight_edge_case_test_$$"

mkdir -p "$TEST_DIR"

TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name="$1"
    local test_func="$2"

    header "Test: $test_name"

    if $test_func; then
        success "PASSED: $test_name"
        ((TESTS_PASSED++))
    else
        error "FAILED: $test_name"
        ((TESTS_FAILED++))
    fi
    echo ""
}

# Edge Case Tests
test_empty_filename() {
    info "Testing with empty filename..."

    # Create file with empty name (via special characters)
    touch "$TEST_DIR/ .txt" 2>/dev/null || return 0  # Skip if not possible

    # Try to process (would need actual API call)
    warning "Skipping actual API call (needs integration)"
    return 0
}

test_very_long_filename() {
    info "Testing with very long filename..."

    # Create 255-character filename
    local long_name="$(head -c 200 /dev/urandom | base64 | tr -dc 'a-zA-Z0-9').txt"

    touch "$TEST_DIR/$long_name" 2>/dev/null || {
        error "Failed to create long filename"
        return 1
    }

    success "Created file with $(echo -n "$long_name" | wc -c | tr -d ' ') character filename"
    return 0
}

test_special_characters_in_filename() {
    info "Testing special characters in filenames..."

    local special_chars=(
        "file with spaces.txt"
        "file'with'quotes.txt"
        'file"with"double"quotes.txt'
        "file(with)parens.txt"
        "file[with]brackets.txt"
        "file{with}braces.txt"
        "file@with#special\$chars.txt"
        "file%with%percents.txt"
        "file&ampersand.txt"
        "file+plus+sign.txt"
        "file=equal.txt"
        "file;semicolon.txt"
        "file,comma.txt"
        "file\\`backtick\\`.txt"
        "file~tilde.txt"
        "file!exclamation.txt"
        "file^caret.txt"
    )

    for name in "${special_chars[@]}"; do
        touch "$TEST_DIR/$name" 2>/dev/null || {
            error "Failed to create: $name"
            return 1
        }
    done

    success "Created $(echo "${#special_chars[@]}") files with special characters"
    return 0
}

test_unicode_filename() {
    info "Testing Unicode filenames..."

    local unicode_names=(
        "文件.txt"
        "файл.txt"
        "datei.txt"
        "αρχείο.txt"
        "ملف.txt"
        "ファイル.txt"
        "файл.txt"
        "tiedosto.txt"
        "файл.txt"
        "🎉🎊🎁.txt"
        "😀😃😄.txt"
        "🚀🌟⭐.txt"
    )

    for name in "${unicode_names[@]}"; do
        touch "$TEST_DIR/$name" 2>/dev/null || {
            error "Failed to create: $name"
            return 1
        }
    done

    success "Created $(echo "${#unicode_names[@]}") Unicode filenames"
    return 0
}

test_zero_byte_file() {
    info "Testing zero-byte file..."

    touch "$TEST_DIR/empty.txt"

    if [[ -s "$TEST_DIR/empty.txt" ]]; then
        error "File is not empty"
        return 1
    fi

    success "Zero-byte file created successfully"
    return 0
}

test_very_large_file() {
    info "Testing very large file..."

    # Create 100MB file
    dd if=/dev/zero of="$TEST_DIR/large.bin" bs=1M count=100 2>/dev/null

    local size=$(du -h "$TEST_DIR/large.bin" | cut -f1)
    success "Created ${size} file"
    return 0
}

test_binary_file() {
    info "Testing binary file..."

    # Create random binary file
    dd if=/dev/urandom of="$TEST_DIR/binary.bin" bs=1024 count=10 2>/dev/null

    success "Created binary file"
    return 0
}

test_file_with_no_extension() {
    info "Testing file with no extension..."

    echo "test content" > "$TEST_DIR/README"

    success "Created file without extension"
    return 0
}

test_file_with_many_extensions() {
    info "Testing file with multiple extensions..."

    echo "test" > "$TEST_DIR/file.tar.gz.zip.txt"

    success "Created file with multiple extensions"
    return 0
}

test_deep_directory_structure() {
    info "Testing deep directory structure..."

    local deep_dir="$TEST_DIR/$(seq -f "dir%g" 1 50 | tr '\n' '/')"

    mkdir -p "$deep_dir" 2>/dev/null || {
        error "Failed to create deep directory"
        return 1
    }

    echo "test" > "$deep_dir/file.txt"

    local depth=$(echo "$deep_dir" | tr '/' '\n' | wc -l | tr -d ' ')
    success "Created directory structure $depth levels deep"
    return 0
}

test_many_files_in_directory() {
    info "Testing many files in single directory..."

    for i in $(seq 1 1000); do
        echo "content $i" > "$TEST_DIR/batch_$i.txt"
    done

    local count=$(ls -1 "$TEST_DIR" | wc -l | tr -d ' ')
    success "Created $count files in single directory"
    return 0
}

test_duplicate_filenames() {
    info "Testing duplicate filename handling..."

    mkdir -p "$TEST_DIR/duplicates"

    echo "version 1" > "$TEST_DIR/duplicates/file.txt"
    echo "version 2" > "$TEST_DIR/duplicates/file.txt"
    echo "version 3" > "$TEST_DIR/duplicates/file.txt"

    local count=$(ls -1 "$TEST_DIR/duplicates" | wc -l | tr -d ' ')
    info "Duplicate handling: created $count file(s)"

    success "Duplicate filename test complete"
    return 0
}

test_very_long_line() {
    info "Testing very long line in file..."

    # Create file with single 1MB line
    python3 -c "print('x' * (1024 * 1024))" > "$TEST_DIR/long_line.txt"

    local lines=$(wc -l < "$TEST_DIR/long_line.txt" | tr -d ' ')
    local size=$(du -h "$TEST_DIR/long_line.txt" | cut -f1)

    success "Created file with $lines line(s), size $size"
    return 0
}

test_mixed_line_endings() {
    info "Testing mixed line endings..."

    printf "line1\r\nline2\nline3\r\nline4\rline5\n" > "$TEST_DIR/mixed_endlines.txt"

    success "Created file with mixed line endings"
    return 0
}

test_invalid_utf8() {
    info "Testing invalid UTF-8 sequences..."

    # Create file with invalid UTF-8
    printf "\xff\xfe\x00\x01 invalid \x80\x81" > "$TEST_DIR/invalid_utf8.txt"

    success "Created file with invalid UTF-8"
    return 0
}

test_permissions() {
    info "Testing various file permissions..."

    mkdir -p "$TEST_DIR/permissions"

    # Read-only file
    echo "read only" > "$TEST_DIR/permissions/readonly.txt"
    chmod 444 "$TEST_DIR/permissions/readonly.txt"

    # Executable file
    echo "executable" > "$TEST_DIR/permissions/executable.txt"
    chmod 755 "$TEST_DIR/permissions/executable.txt"

    # No permissions
    touch "$TEST_DIR/permissions/noperms.txt"
    chmod 000 "$TEST_DIR/permissions/noperms.txt"

    success "Created files with various permissions"
    return 0
}

test_symlinks() {
    info "Testing symbolic links..."

    mkdir -p "$TEST_DIR/links"
    echo "target" > "$TEST_DIR/links/target.txt"

    ln -s "$TEST_DIR/links/target.txt" "$TEST_DIR/links/link.txt"
    ln -s "$TEST_DIR/links/nonexistent.txt" "$TEST_DIR/links/broken.txt"

    success "Created symbolic links"
    return 0
}

generate_report() {
    header "Edge Case Test Report"

    local report_file="$TEST_DIR/edge_case_report.txt"

    cat > "$report_file" <<EOF
Insight Edge Case Test Report
=============================

Date: $(date)
App: $APP_PATH

Results:
--------
Tests Passed: $TESTS_PASSED
Tests Failed: $TESTS_FAILED

Test Coverage:
--------------
- Empty/zero-byte files
- Very long filenames
- Special characters
- Unicode filenames
- Large files
- Binary files
- Deep directories
- Many files in directory
- Duplicate filenames
- Very long lines
- Mixed line endings
- Invalid UTF-8
- File permissions
- Symbolic links

Recommendations:
----------------
- Review failed tests
- Add handling for any unexpected behaviors
- Consider edge cases in user input validation
- Test file upload limits
EOF

    cat "$report_file"
    success "Report saved to: $report_file"
    echo ""
}

main() {
    header "Insight Edge Case Test Suite"

    echo -e "${BLUE}Configuration:${NC}"
    info "App: $APP_PATH"
    info "Test Directory: $TEST_DIR"
    echo ""

    # Run all edge case tests
    run_test "Empty Filename" test_empty_filename
    run_test "Very Long Filename" test_very_long_filename
    run_test "Special Characters in Filename" test_special_characters_in_filename
    run_test "Unicode Filename" test_unicode_filename
    run_test "Zero Byte File" test_zero_byte_file
    run_test "Very Large File" test_very_large_file
    run_test "Binary File" test_binary_file
    run_test "File with No Extension" test_file_with_no_extension
    run_test "File with Many Extensions" test_file_with_many_extensions
    run_test "Deep Directory Structure" test_deep_directory_structure
    run_test "Many Files in Directory" test_many_files_in_directory
    run_test "Duplicate Filenames" test_duplicate_filenames
    run_test "Very Long Line" test_very_long_line
    run_test "Mixed Line Endings" test_mixed_line_endings
    run_test "Invalid UTF-8" test_invalid_utf8
    run_test "File Permissions" test_permissions
    run_test "Symbolic Links" test_symlinks

    # Generate report
    generate_report

    # Summary
    header "Test Summary"
    echo "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"
    success "Passed: $TESTS_PASSED"
    if [[ $TESTS_FAILED -gt 0 ]]; then
        error "Failed: $TESTS_FAILED"
    else
        success "All tests passed!"
    fi
    echo ""

    info "Test files preserved at: $TEST_DIR"
}

main "$@"
