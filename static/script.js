document.addEventListener("DOMContentLoaded", function() {
    // Selecciona el botón por su clase
    var myButton = document.querySelector(".button");

    // Arreglo para almacenar las nuevas características del paciente
    let newpat = [];
    // Arreglo con los nombres de las clases de los campos de características
    let names = [".polyuria",".polydipsia",".weight",".weakness",".polyphagia",".thrush",".blurring",".itching",".irritability",".healing",".paresis",".stiffness",".alopecia",".obesity"]

    myButton.addEventListener("click", async function(event) {
        event.preventDefault(); // Evita el envío del formulario por defecto
        
        // Obtén los valores de edad y género
        let age = document.querySelector(".age").value;
        let gender = document.querySelector(".gender").value;
        newpat = [Number(age), Number(gender)];

        // Itera sobre los nombres y agrega los valores al arreglo newpat
        for (let i = 0; i < names.length; i++) {
            let Var = document.querySelector(names[i]).value;
            newpat.push(Number(Var));
        }

        console.log(newpat);
        

        const API = 'http://127.0.0.1:8800/predict'

        function postData(urlAPI,data){
            const response = fetch(urlAPI, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ features: data })
            })

            return response;
        }

        postData(API,newpat)
            .then(response => response.json())
            .then(data => console.log(`los datos recolectados son ${data.result}`)) 
            .then(data => {
                // Muestra el resultado de la predicción en el HTML
                result = document.querySelector('.result')
                result.innerText = 'La predicción es: ' + data.result;
            })
            .catch(error => console.error('Error buscar los datos:', error));
    });
});

       