document.addEventListener("DOMContentLoaded", function() {
    // Selecciona el botón por su ID
    var myButton = document.querySelector(".button");


    let newpat = []
    let names = [".polyuria",".polydipsia",".weight",".weakness",".polyphagia",".thrush",".blurring",".itching",".irritability",".healing",".paresis",".stiffness",".alopecia",".obesity"]
    let Var;
    myButton.addEventListener("click", async function() {
        let edad = document.querySelector(".age").value
        let gender = document.querySelector(".gender").value
        newpat = [edad,gender]
        
        for(let i=0; i<names.length;i++){
            Var = document.querySelector(names[i]).value
            newpat.push(Var)
        }
    });
});