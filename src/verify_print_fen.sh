while read fen; do
    cutFen=${fen%\ [*}
    output=$((
    echo position fen $cutFen
    echo print_fen
    ) | ./Halogen.exe | tail -1)
    
    if [[ "$cutFen" != "$output" ]]; then
        echo $cutFen
        echo $output
    fi
done < $1




